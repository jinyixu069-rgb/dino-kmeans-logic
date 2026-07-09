# cache_sam3_foreground_masks.py
# 缓存 SAM3 前景 mask,给 k-means 词袋去噪用(而不是做实例计数统计)。
#
# 做的事:对每张图,用文本提示分割出所有匹配实例,取【并集】。
#   mode='foreground': 并集直接就是前景 mask(比如 pushpins 的 "pushpin")。
#   mode='background': 并集是背景/容器 mask,取反后才是前景(比如 breakfast_box
#                       的 "white tray"+"black background" 取反得到全部食物)。
# 再各向异性缩到 48x48(和 DINO anisotropic-672 预处理对齐)得到软覆盖率权重,存盘。
# 这份权重给下游 dual_branch_bow_pipeline.py / triple_branch_bow_pipeline.py 用来:
#   1) 筛选哪些 patch 能参与 k-means 拟合(词表去噪)
#   2) 给直方图统计加权(计数去噪)
# 不做任何打分,不产出 AUROC —— 纯预处理缓存,类似 features_cache 的地位。
#
# SAM3 加载/推理接口照抄 sam3_segment_check.py,没有自己发明 API。
#
# 用法:
#   python cache_sam3_foreground_masks.py --checkpoint /path/to/sam3.pt --obj pushpins
#   python cache_sam3_foreground_masks.py --obj breakfast_box --use-generated-prompts

import os
import sys
import json
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image

DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
DEFAULT_SAM3_ROOT = os.path.join(PROJECT_ROOT, "third_party", "sam3")
DEFAULT_MASK_CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache_fgmask")

GRID = 48  # 672 / 14,和 DINO 特征 patch 排列对齐

# 前景覆盖率闸门:目标是"前景 vs 背景",所以并集 mask 的覆盖率既不该太高也不该太低。
# 太高(>HIGH):基本可以判定 prompt 把容器/背景也盖进去了(比如 breakfast_box 的 "box"),
#              前景分支会退化成"带噪声的全图词袋",这次实测 breakfast_box B 掉到 0.68 就是这个原因。
# 太低(<LOW):前景没盖全(比如 screw_bag 只框到 2 个螺丝漏了螺母垫圈),需要补 prompt。
# 这两个阈值是拍的经验值,只用来"在批量跑之前给出警告",不改变任何缓存结果,不做自动删除。
FG_COVERAGE_HIGH = 0.70
FG_COVERAGE_LOW = 0.02

SPLITS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}

# 只有 pushpins 的提示词是你已经用 sam3_segment_check.py 肉眼确认过的。
# 其它类别先跑 sam3_segment_check.py 确认分割质量,再把验证过的提示词加进这里。
# 值统一用列表:多组件类别(比如 screw_bag)可能需要好几个短语才能覆盖所有前景部件,
# SAM3 一次只处理一个概念,这里对每个短语各查一次 SAM3,取并集。
VALIDATED_PROMPTS = {
    "pushpins": ["pushpin"],
}

GENERATED_PROMPTS_PATH = os.path.join(
    PROJECT_ROOT, "results", "qwen_sam3_prompts", "generated_prompts.json")


def resolve_checkpoint(path):
    if path is None:
        path = os.environ.get("SAM3_CHECKPOINT")
    if path is None:
        raise ValueError("请通过 --checkpoint 或环境变量 SAM3_CHECKPOINT 指定 sam3.pt")
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def import_sam3(sam3_root):
    sam3_root = os.path.abspath(sam3_root)
    if not os.path.isdir(sam3_root):
        raise FileNotFoundError(f"SAM3 repo not found: {sam3_root}")
    sys.path.insert(0, sam3_root)
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    return build_sam3_image_model, Sam3Processor, bpe_path


def inference_context(device):
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def segment_union_mask(processor, image, prompts, device):
    """对 prompts 列表里每个短语各查一次 SAM3,取所有短语、所有实例的并集 mask,
    实例数量按短语累加(比如 screw_bag 用 ["screw","nut","washer"],就是三次查询
    各自的实例数相加)。返回 (原图分辨率 bool mask [H,W], 总实例数量 n_instances)。
    没检测到任何实例时返回全 False mask 和 n_instances=0。"""
    union = None
    total_n = 0
    for prompt in prompts:
        with inference_context(device):
            state = processor.set_image(image)
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(prompt=prompt, state=state)

        masks = state.get("masks")
        if masks is None or masks.shape[0] == 0:
            continue

        masks_np = masks.detach().cpu().numpy()
        if masks_np.ndim == 4:  # [N,1,H,W] -> [N,H,W]
            masks_np = masks_np[:, 0]
        masks_np = masks_np > 0.5
        total_n += int(masks_np.shape[0])
        this_union = masks_np.any(axis=0)
        union = this_union if union is None else (union | this_union)

    if union is None:
        union = np.zeros((image.height, image.width), dtype=bool)
    return union, total_n


def mask_to_grid_weights(mask_2d, grid=GRID):
    """各向异性缩到 grid x grid 的软覆盖率权重 [grid*grid],行优先展平,和 DINO patch 排列对齐。"""
    m = Image.fromarray((mask_2d.astype(np.uint8) * 255))
    m = m.resize((grid, grid), Image.BILINEAR)
    w = np.asarray(m, dtype=np.float32) / 255.0
    return w.reshape(-1)


def list_images(obj, split_subdir):
    d = os.path.join(DATASET_ROOT, obj, split_subdir)
    if not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    fs.sort()
    return fs


def process_split(processor, obj, split_name, split_subdir, prompts, mode, device,
                  mask_cache_dir, overwrite):
    out_dir = os.path.join(mask_cache_dir, obj, split_name)
    os.makedirs(out_dir, exist_ok=True)
    files = list_images(obj, split_subdir)

    coverage = []  # 每张图前景权重占比(已经过 mode 处理,统一是"前景"含义)
    counts = []    # SAM3 检出的实例数量(所有 prompt 累加)。mode='background' 时这个数字
                   # 是"背景材料的实例数",不是前景内容物数量,不能直接喂给分支 C,
                   # 只是留着做诊断参考,下面会打印明确提醒。
    for f in files:
        basename = f.rsplit(".", 1)[0]
        out_path = os.path.join(out_dir, basename + ".npy")
        count_path = os.path.join(out_dir, basename + "_count.npy")
        if not overwrite and os.path.exists(out_path) and os.path.exists(count_path):
            w = np.load(out_path)
            n_inst = int(np.load(count_path))
        else:
            img_path = os.path.join(DATASET_ROOT, obj, split_subdir, f)
            image = Image.open(img_path).convert("RGB")
            mask, n_inst = segment_union_mask(processor, image, prompts, device)
            if mode == "background":
                mask = ~mask  # 取反:背景 mask -> 前景 mask,这是这个范式的核心一步
            w = mask_to_grid_weights(mask)
            np.save(out_path, w.astype(np.float32))
            np.save(count_path, np.array(n_inst, dtype=np.int32))

        cov = float(w.mean())
        coverage.append(cov)
        counts.append(n_inst)
        if cov < 1e-3:
            print(f"  [WARN] {obj}/{split_name}/{basename}: 前景覆盖率≈0,"
                  f"该图 SAM3 可能没检出任何 {prompts} 实例", flush=True)

    if coverage:
        cov = np.array(coverage)
        cnt = np.array(counts)
        mean_cov = cov.mean()
        print(f"  [{obj}/{split_name}] n={len(cov)}  "
              f"前景覆盖率(mode={mode}) mean={mean_cov:.3f} min={cov.min():.3f} max={cov.max():.3f}  "
              f"{'背景' if mode == 'background' else '前景'}实例数 mean={cnt.mean():.2f} "
              f"median={np.median(cnt):.1f} min={cnt.min()} max={cnt.max()}"
              f"{'  [注意: mode=background 时这是背景材料实例数,不是前景内容物数量,分支C不要用]' if mode == 'background' else ''}",
              flush=True)
        # 覆盖率闸门:只在 train_good 上判断,判断的是取反之后的最终前景覆盖率,
        # 所以不管 mode 是 foreground 还是 background,同一套阈值含义一致。
        if split_name == "train_good":
            if mean_cov > FG_COVERAGE_HIGH:
                print(f"  [闸门警告] {obj}: train_good 前景覆盖率 {mean_cov:.3f} > {FG_COVERAGE_HIGH},"
                      f"前景≈全图,{'背景 prompt 可能没盖住真正的背景区域' if mode=='background' else '前景 prompt 很可能把容器/背景也盖进去了'},"
                      f"前景分支会退化成带噪声的全图词袋。", flush=True)
            elif mean_cov < FG_COVERAGE_LOW:
                print(f"  [闸门警告] {obj}: train_good 前景覆盖率 {mean_cov:.3f} < {FG_COVERAGE_LOW},"
                      f"前景基本没被盖住,{'背景 prompt 可能把前景内容物也当成背景框住了' if mode=='background' else 'prompt 可能分不到目标'},"
                      f"检查分割质量或调整 prompt。", flush=True)
    return coverage, counts


def resolve_prompts(obj, cli_prompts, use_generated):
    """按优先级解析该类别要用的 (prompt 列表, mode),并打印清楚来源可信度:
    1. --prompts 显式指定(用户自己确认过,最高优先级,视为直接前景 mode='foreground')
    2. VALIDATED_PROMPTS 里人工用 sam3_segment_check.py 肉眼验证过的(mode='foreground')
    3. --use-generated-prompts 时,回退到 Qwen3-VL 自动生成、未经人工验证的 manifest
       (mode='background',需要下游取反才是前景)
    3 号来源会打印醒目警告,不会假装它和人工验证的一样可信。"""
    if cli_prompts:
        print(f"[{obj}] 使用命令行显式指定的 prompt(直接前景): {cli_prompts}", flush=True)
        return cli_prompts, "foreground"

    if use_generated:
        if not os.path.exists(GENERATED_PROMPTS_PATH):
            raise SystemExit(
                f"[{obj}] 没有人工验证过的 prompt,且找不到自动生成的 manifest: "
                f"{GENERATED_PROMPTS_PATH}\n先跑 generate_sam3_prompts_with_qwen.py。")
        with open(GENERATED_PROMPTS_PATH) as f:
            manifest = json.load(f)
        if obj not in manifest:
            raise SystemExit(f"[{obj}] 不在自动生成的 manifest 里,该类别的 prompt 生成失败过,"
                             f"需要单独处理: {GENERATED_PROMPTS_PATH}")
        entry = manifest[obj]
        prompts, mode = entry["phrases"], entry.get("mode", "background")
        print(f"[{obj}] [WARN] 使用 Qwen3-VL 自动生成、尚未人工验证的背景描述 "
              f"(mode={mode}): {prompts}\n"
              f"  下游会把这些短语的并集 mask 取反当作前景。强烈建议先用 "
              f"sam3_segment_check.py 肉眼确认背景真的被分割干净、没有连带框住前景内容物,"
              f"再信任这批缓存结果。", flush=True)
        return prompts, mode

    if obj in VALIDATED_PROMPTS:
        prompts = VALIDATED_PROMPTS[obj]
        print(f"[{obj}] 使用人工已验证的 prompt(直接前景): {prompts}", flush=True)
        return prompts, "foreground"

    raise SystemExit(
        f"[{obj}] 没有已验证的提示词。可以: (1) 用 --prompts 显式指定并手动验证过;"
        f"(2) 先跑 sam3_segment_check.py 肉眼确认后加进 VALIDATED_PROMPTS;"
        f"(3) 加 --use-generated-prompts 使用 Qwen3-VL 自动生成的(未验证,风险自负)。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pushpins",
                    choices=["breakfast_box", "juice_bottle", "pushpins",
                             "screw_bag", "splicing_connectors"])
    ap.add_argument("--prompts", nargs="+", default=None,
                    help="显式指定一个或多个文本提示词(空格分隔)。不给则按"
                         "VALIDATED_PROMPTS -> --use-generated-prompts 的顺序解析")
    ap.add_argument("--mode", choices=["foreground", "background"], default=None,
                    help="--prompts 显式指定时,配套说明这些短语是直接前景还是背景"
                         "(背景会取反)。不给默认 foreground(直接前景,兼容旧用法)")
    ap.add_argument("--use-generated-prompts", action="store_true",
                    help="没有人工验证过的 prompt 时,回退到 Qwen3-VL 自动生成的"
                         "manifest(未经人工验证,建议先跑 sam3_segment_check.py 复核)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sam3-root", default=DEFAULT_SAM3_ROOT)
    ap.add_argument("--mask-cache-dir", default=DEFAULT_MASK_CACHE_DIR)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.prompts:
        prompts, mode = args.prompts, (args.mode or "foreground")
        print(f"[{args.obj}] 使用命令行显式指定的 prompt(mode={mode}): {prompts}", flush=True)
    else:
        prompts, mode = resolve_prompts(args.obj, None, args.use_generated_prompts)

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"

    build_sam3_image_model, Sam3Processor, bpe_path = import_sam3(args.sam3_root)
    print(f"Loading SAM3 from {checkpoint} on {device}, prompts={prompts}, mode={mode} ...",
          flush=True)
    model = build_sam3_image_model(
        bpe_path=bpe_path, checkpoint_path=checkpoint,
        load_from_HF=False, device=device,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence)

    for split_name, split_subdir in SPLITS.items():
        print(f"\n[{args.obj}/{split_name}]", flush=True)
        process_split(processor, args.obj, split_name, split_subdir, prompts, mode, device,
                      args.mask_cache_dir, args.overwrite)

    print(f"\nDone. Foreground weight masks cached under "
          f"{os.path.join(args.mask_cache_dir, args.obj)}/<split>/<basename>.npy, "
          f"instance counts cached alongside as <basename>_count.npy", flush=True)


if __name__ == "__main__":
    main()
