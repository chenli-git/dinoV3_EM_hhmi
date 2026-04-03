"""
Task 2: Feature Extraction with DINOv3

Patch size (Task 2.1):
    ViT-B/16 uses 16x16 px patches. At s1 (8 nm/px) a mitochondrion is ~100 px
    across — roughly 6 patches — so each patch captures sub-structure.

Dense embeddings (Task 2.2):
    One token per 16x16 patch. Bilinear upsample from (H/16, W/16) back to (H, W)
    gives dense per-pixel embeddings with no learned parameters.
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from dotenv import load_dotenv
from transformers import AutoModel

load_dotenv()

DATA_DIR = "data"
MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
PATCH_SIZE = 16
DATASETS = ["hela-2", "hela-3"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMG_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def load_backbone():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model = AutoModel.from_pretrained(MODEL_ID, token=token)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False

    n_reg = getattr(model.config, "num_register_tokens", 0)
    n_prefix = 1 + n_reg  # CLS + register tokens
    feat_dim = model.config.hidden_size

    print(f"  model: {MODEL_ID}")
    print(f"  patch_size={model.config.patch_size}  feat_dim={feat_dim}")
    print(f"  prefix tokens stripped: {n_prefix} (1 CLS + {n_reg} register)")
    return model, n_prefix, feat_dim


def normalize(arr_uint16):
    lo, hi = np.percentile(arr_uint16, [1, 99])
    arr = np.clip(arr_uint16, lo, hi).astype(np.float32)
    return (arr - lo) / (hi - lo + 1e-6)


def to_tensor(img):
    """[H,W] float32 -> [1,3,H,W] ImageNet-normalized, padded to patch multiple."""
    H, W = img.shape
    pH = math.ceil(H / PATCH_SIZE) * PATCH_SIZE
    pW = math.ceil(W / PATCH_SIZE) * PATCH_SIZE

    x = torch.from_numpy(img[None, None]).to(DEVICE).repeat(1, 3, 1, 1)
    if pH != H or pW != W:
        x = F.pad(x, (0, pW - W, 0, pH - H))
    x = (x - IMG_MEAN) / IMG_STD
    return x, H, W


def encode_slice(backbone, n_prefix, feat_dim, img):
    x, orig_H, orig_W = to_tensor(img)
    _, _, pH, pW = x.shape
    grid_h, grid_w = pH // PATCH_SIZE, pW // PATCH_SIZE

    with torch.no_grad():
        tokens = backbone(pixel_values=x).last_hidden_state[:, n_prefix:, :]  # [1, N, 768]

    patch_tokens = tokens[0].cpu().numpy()  # [N, 768]
    feat_map = (tokens
                .transpose(1, 2)
                .reshape(1, feat_dim, grid_h, grid_w)
                [0].cpu().numpy())  # [768, gh, gw]

    return patch_tokens, feat_map


def get_dense(feat_map, H, W):
    """Upsample patch-grid features to full image resolution."""
    t = torch.from_numpy(feat_map[None]).to(DEVICE)
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0].cpu().numpy()  # [768, H, W]


def process_dataset(name, backbone, n_prefix, feat_dim):
    print(f"\n--- {name} ---")

    em = zarr.open(os.path.join(DATA_DIR, name, "em.zarr"), "r")
    print(f"  zarr shape: {em.shape}  dtype={em.dtype}")

    Z, H, W = em.shape
    grid_h = math.ceil(H / PATCH_SIZE)
    grid_w = math.ceil(W / PATCH_SIZE)
    N = grid_h * grid_w
    print(f"  slices={Z}  size={H}x{W}  grid={grid_h}x{grid_w}  N={N}")

    patch_tokens = np.zeros((Z, N, feat_dim), dtype=np.float32)
    feat_maps    = np.zeros((Z, feat_dim, grid_h, grid_w), dtype=np.float32)

    dense_path = os.path.join(DATA_DIR, name, "dense_embeddings.zarr")
    dense_zarr = zarr.open(
        dense_path, mode="w",
        shape=(Z, feat_dim, H, W), chunks=(1, 64, H, W),
        dtype="float16",
    )

    for i in range(Z):
        img = normalize(em[i])
        pt, fm = encode_slice(backbone, n_prefix, feat_dim, img)
        patch_tokens[i] = pt
        feat_maps[i] = fm
        dense_zarr[i] = get_dense(fm, H, W).astype(np.float16)
        print(f"  slice {i+1}/{Z}", end="\r")
    print()

    out_path = os.path.join(DATA_DIR, name, "embeddings.npz")
    np.savez_compressed(
        out_path,
        patch_tokens=patch_tokens,
        feat_maps=feat_maps,
        grid_shape=np.array([grid_h, grid_w]),
        orig_shape=np.array([H, W]),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  saved embeddings.npz  ({size_mb:.0f} MB)")
    print(f"    patch_tokens: {patch_tokens.shape}")
    print(f"    feat_maps:    {feat_maps.shape}")
    print(f"    dense:        {dense_zarr.shape}  float16  -> {dense_path}")


def main():
    print(f"device: {DEVICE}")
    print("loading DINOv3...")
    backbone, n_prefix, feat_dim = load_backbone()

    for name in DATASETS:
        process_dataset(name, backbone, n_prefix, feat_dim)

    print("\ndone.")


if __name__ == "__main__":
    main()
