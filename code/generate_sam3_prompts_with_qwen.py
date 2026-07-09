# generate_sam3_prompts_with_qwen.py
# 用 Qwen3-VL-8B-Instruct 给每个类别自动生成 SAM3 该用的文本 prompt,
# 替代"每个类别都要人工跑 sam3_segment_check.py 肉眼挑 prompt"这一步。
#
# 关键设计依据(查证过,不是拍脑袋): SAM3 的文本编码器是围绕"短名词短语"
# (noun phrase)训练的,官方例子都是 "yellow school bus"、"striped cat" 这种
# 2-4 词的短语,不是完整句子。Meta 自己的博客也明确说 MLLM 配合 SAM3 使用时,
# 标准做法是让 MLLM 生成的长描述"蒸馏"成短概念短语再喂给 SAM3,不能把一整段
# 场景描述直接丢给它。所以这份脚本不是让 Qwen3-VL"描述图片",而是强约束它
# 只输出一个短语列表,格式直接对齐 SAM3 的输入习惯。
#
# 免费自检: pushpins 你已经人工验证过 prompt="pushpin"(sam3_segment_check.py)。
# 这份脚本也会对 pushpins 跑一遍自动生成,和已验证的人工结果做对比打印出来——
# 如果自动生成的结果和人工验证的一致/接近,说明这套自动化流程本身是可信的;
# 如果对不上,说明自动化这条路本身有问题,不该先急着用它去处理其它 4 个类别。
#
# 产出: 一份 JSON manifest,每个类别对应一个"短语列表"(不是单个字符串,
# 因为像 breakfast_box、screw_bag 这种多组件类别,可能需要好几个短语才能
# 覆盖所有前景部件类型,SAM3 一次只处理一个概念)。
#
# 重要: 这份 manifest 是"未经人工验证"的产出,不能直接当成 VALIDATED_PROMPTS
# 那样的可信来源使用——用之前必须先对每个类别至少跑一次 sam3_segment_check.py
# 肉眼确认分割质量,这是你自己定的验证流程,这份脚本不改变这一点。
#
# 用法:
#   python generate_sam3_prompts_with_qwen.py --n-ref-images 2
#   python generate_sam3_prompts_with_qwen.py --obj breakfast_box juice_bottle

import os
import json
import argparse
import re
import random

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
DEFAULT_MODEL_PATH = "/mnt/nfs/xujy/logicdataset/models/Qwen3-VL-8B-Instruct"
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "qwen_sam3_prompts")

ALL_OBJS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

# 你已经人工用 sam3_segment_check.py 肉眼验证过的 prompt,只用来做自检对比,
# 不会被这份脚本覆盖或替代。
HUMAN_VALIDATED = {
    "pushpins": ["pushpin"],
}

# 目标已明确:前景 vs 背景。前景分支只需要把"该关注的东西"和背景(黑底/托盘/网格/塑料袋)
# 分开,不追求部件级精细分割。所以这里让 Qwen 列出"能盖全前景的物件词",可以是多个词
# (下游取并集盖全前景),但明确禁止输出会把背景也一起盖进去的【容器整体词】——
# 比如 breakfast_box 的正确答案是里面的食物(orange/peach/granola/nut),不是 "box"(会连托盘一起盖掉)。
FORBIDDEN_WHOLE_CONTAINER_WORDS = {
    "box", "tray", "container", "bag", "package", "packaging", "plate",
    "carton", "case", "background", "surface", "grid", "mesh", "plastic bag",
}

SYSTEM_PROMPT = (
    "You are a vision assistant configuring an open-vocabulary segmentation model (SAM3) "
    "for an industrial visual-inspection pipeline. SAM3 accepts short English noun phrases "
    "(2-4 words each, e.g. \"pushpin\", \"screw\", \"orange fruit\") -- it does NOT understand "
    "full sentences.\n"
    "GOAL: separate the meaningful FOREGROUND CONTENTS from the background. The background "
    "includes the black backdrop, the tray/box/bag/container that merely HOLDS the contents, "
    "and any support surface or mesh. These must NOT be segmented.\n"
    "Instead, list the noun phrases for the actual CONTENT items placed in/on the product "
    "(e.g. the individual food items inside a tray, the individual hardware parts inside a bag). "
    "Choose phrases that together COVER ALL the foreground contents -- use several phrases if "
    "there are several distinct content types.\n"
    "STRICT RULE: never output a word for the holder/container itself (box, tray, bag, "
    "container, plate, background, surface, mesh). If the product is a bottle or jar whose "
    "body IS the inspected object (nothing meaningful inside to separate), then the object "
    "name itself is acceptable.\n"
    "Respond with ONLY a JSON array of short noun phrases, no explanation, no markdown fences.\n"
    "Example (breakfast tray): [\"orange\", \"peach\", \"granola\", \"nut\"]\n"
    "Example (hardware bag): [\"screw\", \"nut\", \"washer\"]"
)

USER_PROMPT = (
    "These images show a normal (defect-free) sample of one product category. "
    "List the foreground CONTENT item types (not the holder/container/background) as a "
    "JSON array of short noun phrases that together cover all the foreground."
)


def load_model(model_path, device):
    print(f"Loading Qwen3-VL from {model_path} ...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto" if device == "auto" else device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def sample_reference_images(obj, n, seed=0):
    d = os.path.join(DATASET_ROOT, obj, "train", "good")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise FileNotFoundError(f"没有在 {d} 找到 train/good 图片")
    rng = random.Random(seed)
    picked = rng.sample(files, min(n, len(files)))
    return [Image.open(os.path.join(d, f)).convert("RGB") for f in picked], picked


def build_messages(images):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": USER_PROMPT})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


@torch.no_grad()
def query_qwen(model, processor, images, max_new_tokens=128):
    messages = build_messages(images)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # 只取新生成的部分,不含输入 prompt 本身
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return text.strip()


def parse_phrase_list(raw_text):
    """从模型输出里稳健地解析出一个短语列表。模型即使被强约束,也可能包裹一层
    markdown 代码块或加几句解释,这里做容错,解析不出来就返回 None 而不是硬拼凑。"""
    text = raw_text.strip()
    # 去掉可能的 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()

    # 直接尝试整体解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [p.strip() for p in parsed if p.strip()]
    except json.JSONDecodeError:
        pass

    # 退一步:从文本里找第一个 [...] 片段再解析
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [p.strip() for p in parsed if p.strip()]
        except json.JSONDecodeError:
            pass

    return None


def filter_container_words(phrases):
    """去掉会把背景一起盖住的容器整体词(box/tray/bag/...)。返回 (保留的短语, 被去掉的短语)。
    这是对 prompt 生成的第一道兜底:即使 Qwen 没听话给了 'box',也不会让它进入 SAM3。"""
    kept, dropped = [], []
    for p in phrases:
        if p.strip().lower() in FORBIDDEN_WHOLE_CONTAINER_WORDS:
            dropped.append(p)
        else:
            kept.append(p)
    return kept, dropped


def generate_for_object(model, processor, obj, n_ref_images, seed, max_new_tokens):
    images, picked_files = sample_reference_images(obj, n_ref_images, seed)
    print(f"\n[{obj}] 参考图: {picked_files}", flush=True)

    raw = query_qwen(model, processor, images, max_new_tokens)
    print(f"  原始输出: {raw}", flush=True)

    phrases = parse_phrase_list(raw)
    if phrases is None:
        print(f"  [WARN] 解析失败,无法从输出里提取 JSON 短语列表,该类别跳过,"
              f"需要人工检查上面的原始输出、调整 prompt 或换更大的模型", flush=True)
        return None

    phrases, dropped = filter_container_words(phrases)
    if dropped:
        print(f"  [WARN] 已过滤掉容器整体词(会连背景一起盖住): {dropped}", flush=True)
    if not phrases:
        print(f"  [WARN] 过滤后没有可用的前景短语了(Qwen 只给了容器词),该类别跳过,"
              f"需要人工检查原始输出或调整 prompt", flush=True)
        return None

    print(f"  解析出的短语: {phrases}", flush=True)

    if obj in HUMAN_VALIDATED:
        human = HUMAN_VALIDATED[obj]
        overlap = set(p.lower() for p in phrases) & set(h.lower() for h in human)
        print(f"  [自检] 人工已验证的 prompt: {human}  "
              f"{'✓ 有重合,自动化结果和人工验证一致' if overlap else '✗ 无重合,需要人工复核这条自动化流程'}",
              flush=True)

    return phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", nargs="+", default=ALL_OBJS, choices=ALL_OBJS)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--n-ref-images", type=int, default=2,
                    help="每个类别喂给 Qwen3-VL 的 train_good 参考图数量,不需要多,这一步很便宜")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, processor = load_model(args.model_path, args.device)

    manifest = {}
    failed = []
    for obj in args.obj:
        phrases = generate_for_object(
            model, processor, obj, args.n_ref_images, args.seed, args.max_new_tokens)
        if phrases is None:
            failed.append(obj)
        else:
            manifest[obj] = phrases

    out_path = os.path.join(args.out_dir, "generated_prompts.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved manifest to {out_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed:
        print(f"\n[WARN] 以下类别解析失败,manifest 里没有它们,需要单独处理: {failed}")
    print(
        "\n重要提醒: 这份 manifest 是 Qwen3-VL 自动生成的,还没有人工肉眼验证过。"
        "\n用它跑 cache_sam3_foreground_masks.py 之前,先对每个类别至少跑一次"
        "\nsam3_segment_check.py 确认分割质量,不要跳过这一步直接批量生产 mask。"
    )


if __name__ == "__main__":
    main()
