# extract_dino_features.py
# 用 DINOv2-with-registers-giant 提取 patch token 特征。
# 关键点:
#   - 各向异性 resize 到 672x672(不裁剪、不 pad),避免中心裁剪切掉 LOCO 非正方形图两端的内容
#   - 去掉 CLS token 和 register token,只保留 patch token
#   - 按 mean layers [-18,-12](闭区间)取平均
#   - 存成 float16 .npy,每张图 [2304, 1536] ≈ 6.75MB
#
# 用法(可对不同类别分卡并行):
#   CUDA_VISIBLE_DEVICES=0 python extract_dino_features.py --obj pushpins
#   CUDA_VISIBLE_DEVICES=1 python extract_dino_features.py --obj breakfast_box
#   ...

import os
import argparse
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

# ---------------- 路径配置 ----------------
DEFAULT_MODEL_PATH = (
    "/home/xjy/.cache/huggingface/hub/"
    "models--facebook--dinov2-with-registers-giant/snapshots/"
    "8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb"
)
DATASET_ROOT = "/mnt/nfs/xujy/logicdataset/datasets/mvtec_loco_anomaly_detection"
PROJECT_ROOT = "/mnt/nfs/xujy/logicdataset/dino_kmeans_logic"
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "features_cache")

# ---------------- 提取配置 ----------------
IMAGE_SIZE = 672
LAYER_START, LAYER_END = -18, -12    # 闭区间,含两端
BATCH_SIZE = 4
DTYPE = torch.bfloat16

# DINOv2 官方标准化常数(已在沙盒中核对)
IMAGENET_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

SPLITS = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_logical": "test/logical_anomalies",
}


def list_images(d):
    if not os.path.isdir(d):
        return []
    fs = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    fs.sort()
    return fs


def preprocess_image(path):
    """各向异性 resize 到 IMAGE_SIZE x IMAGE_SIZE,再做 ImageNet 标准化。返回 [3,H,W] float tensor。"""
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = torch.from_numpy(np.array(img, copy=True)).float().permute(2, 0, 1) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def resolve_layer_indices(num_hidden_states, start_neg, end_neg):
    start = num_hidden_states + start_neg
    end = num_hidden_states + end_neg
    assert 0 <= start <= end < num_hidden_states, (start, end, num_hidden_states)
    return list(range(start, end + 1))


@torch.no_grad()
def extract_batch(model, image_paths, num_register_tokens, layer_indices, device):
    batch = torch.stack([preprocess_image(p) for p in image_paths], dim=0)
    pixel_values = batch.to(device=device, dtype=DTYPE)

    out = model(pixel_values=pixel_values, output_hidden_states=True)
    stacked = torch.stack([out.hidden_states[i] for i in layer_indices], dim=0)  # [L,B,T,D]
    mean_over_layers = stacked.mean(dim=0)  # [B,T,D]

    # token 排列: [CLS, register x R, patch x N] —— 已在沙盒确认此顺序
    patch_tokens = mean_over_layers[:, 1 + num_register_tokens:, :]  # [B,N,D]
    return patch_tokens.to(torch.float16).cpu().numpy()


def process_split(model, obj, split_name, split_subdir,
                  num_register_tokens, layer_indices, device, cache_dir, overwrite):
    src_dir = os.path.join(DATASET_ROOT, obj, split_subdir)
    out_dir = os.path.join(cache_dir, obj, split_name)
    os.makedirs(out_dir, exist_ok=True)

    files = list_images(src_dir)
    if not files:
        print(f"[WARN] no images in {src_dir}, skip")
        return

    todo = [f for f in files
            if overwrite or not os.path.exists(os.path.join(out_dir, f.rsplit(".", 1)[0] + ".npy"))]
    print(f"[{obj}/{split_name}] {len(files)} total, {len(todo)} to extract")

    for i in range(0, len(todo), BATCH_SIZE):
        batch_files = todo[i:i + BATCH_SIZE]
        batch_paths = [os.path.join(src_dir, f) for f in batch_files]
        feats = extract_batch(model, batch_paths, num_register_tokens, layer_indices, device)
        for f, feat in zip(batch_files, feats):
            np.save(os.path.join(out_dir, f.rsplit(".", 1)[0] + ".npy"), feat)
        print(f"  [{obj}/{split_name}] {min(i+BATCH_SIZE, len(todo))}/{len(todo)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True,
                    choices=["breakfast_box", "juice_bottle", "pushpins",
                             "screw_bag", "splicing_connectors"])
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_path} on {device} ...")
    model = AutoModel.from_pretrained(args.model_path, dtype=DTYPE).to(device)
    model.eval()

    num_register_tokens = model.config.num_register_tokens
    num_hidden_states = model.config.num_hidden_layers + 1
    layer_indices = resolve_layer_indices(num_hidden_states, LAYER_START, LAYER_END)
    print(f"num_register_tokens={num_register_tokens}, "
          f"num_hidden_layers={model.config.num_hidden_layers}, "
          f"layer indices={layer_indices} (of {num_hidden_states})")

    for split_name, split_subdir in SPLITS.items():
        process_split(model, args.obj, split_name, split_subdir,
                      num_register_tokens, layer_indices, device,
                      args.cache_dir, args.overwrite)

    print(f"Done. Features cached under {os.path.join(args.cache_dir, args.obj)}")


if __name__ == "__main__":
    main()