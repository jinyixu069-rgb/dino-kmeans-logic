# cache_sam3_foreground_masks.py
# 缓存 SAM3 前景 mask,给 k-means 词袋去噪用(而不是做实例计数统计)。
#
# 做的事很简单:对每张图,用文本提示分割出所有匹配实例,取【并集】得到一张前景/背景二值图,
# 各向异性缩到 48x48(和 DINO anisotropic-672 预处理对齐)得到软覆盖率权重,存盘。
# 这份权重只是给下游 run_pipeline.py 用来:
#   1) 筛选哪些 patch 能参与 k-means 拟合(词表去噪)
#   2) 给直方图统计加权(计数去噪)
# 不做任何打分,不产出 AUROC —— 纯预处理缓存,类似 features_cache 的地位。
#
# SAM3 加载/推理接口照抄 sam3_segment_check.py,没有自己发明 API。
#
# 用法:
#   python cache_sam3_foreground_masks.py --checkpoint /path/to/sam3.pt --obj pushpins

import os
import sys
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

SPLITS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}

# 只有 pushpins 的提示词是你已经用 sam3_segment_check.py 肉眼确认过的。
# 其它类别先跑 sam3_segment_check.py 确认分割质量,再把验证过的提示词加进这里。
VALIDATED_PROMPTS = {
    "pushpins": "pushpin",
}


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


def segment_union_mask(processor, image, prompt, device):
    """返回原图分辨率的前景并集 bool mask [H,W](所有匹配实例取并集)。没检测到实例时返回全 False。"""
    with inference_context(device):
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state.get("masks")
    if masks is None or masks.shape[0] == 0:
        return np.zeros((image.height, image.width), dtype=bool)

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4:  # [N,1,H,W] -> [N,H,W]
        masks_np = masks_np[:, 0]
    return (masks_np > 0.5).any(axis=0)


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


def process_split(processor, obj, split_name, split_subdir, prompt, device,
                  mask_cache_dir, overwrite):
    out_dir = os.path.join(mask_cache_dir, obj, split_name)
    os.makedirs(out_dir, exist_ok=True)
    files = list_images(obj, split_subdir)

    coverage = []  # 每张图前景权重占比,用来发现"整图检不到前景"的异常情况
    for f in files:
        basename = f.rsplit(".", 1)[0]
        out_path = os.path.join(out_dir, basename + ".npy")
        if not overwrite and os.path.exists(out_path):
            w = np.load(out_path)
        else:
            img_path = os.path.join(DATASET_ROOT, obj, split_subdir, f)
            image = Image.open(img_path).convert("RGB")
            mask = segment_union_mask(processor, image, prompt, device)
            w = mask_to_grid_weights(mask)
            np.save(out_path, w.astype(np.float32))

        cov = float(w.mean())
        coverage.append(cov)
        if cov < 1e-3:
            print(f"  [WARN] {obj}/{split_name}/{basename}: 前景覆盖率≈0,"
                  f"该图 SAM3 可能没检出任何 '{prompt}' 实例", flush=True)

    if coverage:
        cov = np.array(coverage)
        print(f"  [{obj}/{split_name}] n={len(cov)}  "
              f"前景覆盖率 mean={cov.mean():.3f} min={cov.min():.3f} max={cov.max():.3f}",
              flush=True)
    return coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pushpins",
                    choices=["breakfast_box", "juice_bottle", "pushpins",
                             "screw_bag", "splicing_connectors"])
    ap.add_argument("--prompt", default=None,
                    help="文本提示词。不给则用已验证映射(目前只有 pushpins->'pushpin')")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--sam3-root", default=DEFAULT_SAM3_ROOT)
    ap.add_argument("--mask-cache-dir", default=DEFAULT_MASK_CACHE_DIR)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prompt = args.prompt or VALIDATED_PROMPTS.get(args.obj)
    if prompt is None:
        raise SystemExit(
            f"[{args.obj}] 没有已验证的提示词。先用 sam3_segment_check.py 肉眼确认分割质量,"
            f"验证过后用 --prompt 显式指定。")

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = "cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu"

    build_sam3_image_model, Sam3Processor, bpe_path = import_sam3(args.sam3_root)
    print(f"Loading SAM3 from {checkpoint} on {device}, prompt='{prompt}' ...", flush=True)
    model = build_sam3_image_model(
        bpe_path=bpe_path, checkpoint_path=checkpoint,
        load_from_HF=False, device=device,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.confidence)

    for split_name, split_subdir in SPLITS.items():
        print(f"\n[{args.obj}/{split_name}]", flush=True)
        process_split(processor, args.obj, split_name, split_subdir, prompt, device,
                      args.mask_cache_dir, args.overwrite)

    print(f"\nDone. Foreground weight masks cached under "
          f"{os.path.join(args.mask_cache_dir, args.obj)}/<split>/<basename>.npy", flush=True)


if __name__ == "__main__":
    main()
