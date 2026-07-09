# generate_sam3_prompts_with_qwen.py
# 用 Qwen3-VL-8B-Instruct 给每个类别自动生成 SAM3 该用的文本 prompt,
# 替代"每个类别都要人工跑 sam3_segment_check.py 肉眼挑 prompt"这一步。
#
# 范式(第二版,取代直接列举前景内容物的第一版): 让 Qwen 描述【背景/托底材料】
# (黑背景、白托盘、塑料袋、金属网),下游取反得到前景,而不是直接描述前景内容物。
# 原因是实测教训: 直接列举前景内容物这条路,juice_bottle("cherry juice")和
# splicing_connectors("network cable")都在 SAM3 上失败(0 实例)——这类内容物
# 短语是小众组合词,SAM3 认不出来。而背景材料是极通用的视觉概念,识别可靠得多。
# 这也是 SALAD 论文本身的机制:先定位背景/容器,取反得到前景。
#
# 关键设计依据(查证过,不是拍脑袋): SAM3 的文本编码器是围绕"短名词短语"
# (noun phrase)训练的,官方例子都是 "yellow school bus"、"black background" 这种
# 2-4 词的短语,不是完整句子。所以这份脚本强约束 Qwen 只输出一个短语列表,
# 格式直接对齐 SAM3 的输入习惯。
#
# 自检: pushpins 有人工验证过的前景词 "pushpin"(sam3_segment_check.py)。
# 这份脚本对 pushpins 也跑一遍背景描述生成,检查 Qwen 有没有把 "pushpin" 这个
# 已知前景词错当成背景描述出来——这是范式反转后更有意义的自检方式。
# pushpins 本身仍然用人工验证过的直接前景 prompt(HUMAN_VALIDATED),不走反转。
#
# 产出: 一份 JSON manifest,每个类别对应 {"mode": "background", "phrases": [...]}。
# 下游 cache_sam3_foreground_masks.py 需要读这个 mode 字段,把并集 mask 取反。
#
# 重要: 这份 manifest 是"未经人工验证"的产出,不能直接当成 VALIDATED_PROMPTS
# 那样的可信来源使用——用之前必须先对每个类别至少跑一次 sam3_segment_check.py
# 肉眼确认背景真的被分割干净、没有把前景内容物也框进去。
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
# 不会被这份脚本覆盖或替代。这些是"直接前景"prompt,不走背景反转这条路——
# pushpins 已经验证直接前景效果很好,没必要为了范式统一牺牲已验证的方案。
HUMAN_VALIDATED = {
    "pushpins": ["pushpin"],
}

# 范式改变的原因(实测教训,不是拍脑袋): 直接让 Qwen 列举"前景内容物"这条路对
# juice_bottle("cherry juice"/"orange juice")和 splicing_connectors("network cable")
# 都失败了——SAM3 在 0 实例。这类内容物短语往往是小众组合词,SAM3 文本编码器没有
# 强对应。而背景/托底材料(黑背景、白托盘、塑料袋、金属网)是极其通用的视觉概念,
# 无论是 Qwen 描述还是 SAM3 识别都容易得多。SALAD 论文本身也是这个思路:先定位
# 背景/容器,再取反得到前景,而不是直接对前景内容物做开放词表分割。
# 所以这里让 Qwen 描述背景/托底材料,下游取反得到前景,不再要求穷举内容物种类。
SYSTEM_PROMPT = (
    "You are a vision assistant configuring an open-vocabulary segmentation model (SAM3) "
    "for an industrial visual-inspection pipeline. SAM3 accepts short English noun phrases "
    "(2-4 words each, e.g. \"black background\", \"wire mesh\", \"plastic bag\") -- it does "
    "NOT understand full sentences.\n"
    "GOAL: segment the BACKGROUND / holder so it can be INVERTED to get the foreground "
    "contents (this is more reliable than naming the foreground contents directly, since "
    "foreground content items are often unusual compound nouns that segmentation models "
    "struggle with, while the backdrop/holder material is a common, easy-to-recognize "
    "visual concept).\n"
    "Describe the BACKGROUND / supporting material -- the black backdrop, the tray/bag/box "
    "material that merely HOLDS the contents, or the support surface/mesh underneath -- "
    "NOT the inspected contents themselves. Use several phrases if the background has "
    "multiple distinct materials (e.g. both a black backdrop AND a visible tray/bag).\n"
    "EXCEPTION: if the product itself IS a single self-contained object with nothing "
    "meaningful placed inside/on it to separate (e.g. a bottle, where the bottle body is "
    "the entire inspected object), then only the black backdrop behind it is background -- "
    "do not name the object itself as background.\n"
    "Respond with ONLY a JSON array of short noun phrases, no explanation, no markdown fences.\n"
    "Example (food tray): [\"white tray\", \"black background\"]\n"
    "Example (hardware bag): [\"plastic bag\", \"black background\"]\n"
    "Example (bottle on black backdrop): [\"black background\"]"
)

USER_PROMPT = (
    "These images show a normal (defect-free) sample of one product category. "
    "List the BACKGROUND / holder material types (not the inspected contents) as a "
    "JSON array of short noun phrases, so the background can later be inverted to "
    "isolate the foreground contents."
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

    print(f"  解析出的背景短语: {phrases}", flush=True)

    # 自检:如果该类别有人工验证过的"前景"词(比如 pushpins 的 "pushpin"),
    # 检查 Qwen 有没有把真正的前景内容物错当成背景描述出来——这是范式反转后
    # 更有意义的自检方式,而不是像旧版那样比较"前景词是否重合"。
    if obj in HUMAN_VALIDATED:
        fg_words = set(h.lower() for h in HUMAN_VALIDATED[obj])
        bg_words = set(p.lower() for p in phrases)
        overlap = fg_words & bg_words
        if overlap:
            print(f"  [自检失败] Qwen 把已知前景词 {overlap} 当成背景描述出来了,"
                  f"这条自动化流程对这个类别不可信,需要人工复核。", flush=True)
        else:
            print(f"  [自检通过] 生成的背景短语没有和已知前景词 {fg_words} 冲突。", flush=True)

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
            # mode="background": 下游 cache_sam3_foreground_masks.py 要把这些短语的
            # 并集 mask 取反,才是前景。这是范式核心,不能丢在存盘这一步。
            manifest[obj] = {"mode": "background", "phrases": phrases}

    out_path = os.path.join(args.out_dir, "generated_prompts.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved manifest to {out_path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed:
        print(f"\n[WARN] 以下类别解析失败,manifest 里没有它们,需要单独处理: {failed}")
    print(
        "\n重要提醒: 这份 manifest 是 Qwen3-VL 自动生成的背景描述,还没有人工肉眼验证过,"
        "\n且需要 cache_sam3_foreground_masks.py 取反才能得到前景。用它之前,先对每个类别"
        "\n至少跑一次 sam3_segment_check.py(用这些短语查 SAM3,肉眼确认背景真的被分割干净、"
        "\n没有把前景内容物也框进去),不要跳过这一步直接批量生产 mask。"
    )


if __name__ == "__main__":
    main()
