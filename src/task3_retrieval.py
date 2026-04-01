"""
Task 3: Embedding-Based Retrieval & Visualization

Pipeline:
1. For each (z, mito_id) pair, mean-pool DINOv3 patch tokens within the mito mask
   (seg downsampled to patch-grid resolution). Each slice is an independent 2D sample.
2. Select the largest (z, mito_id) sample in hela-2 as the query.
3. Rank all other mitos by cosine similarity to the query embedding.
4. Visualize:
   - Within-dataset: query vs hela-2 mitos  (task3_within_retrieval.png)
   - Cross-dataset:  query vs hela-3 mitos  (task3_cross_retrieval.png)
"""

import os
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
DATASETS = ["hela-2", "hela-3"]
TOP_K = 5


def load_dataset(name):
    emb = np.load(os.path.join(DATA_DIR, name, "embeddings.npz"))
    feat_maps  = emb["feat_maps"]   # [Z, 768, grid_h, grid_w]
    grid_shape = emb["grid_shape"]  # [grid_h, grid_w]
    seg = zarr.open(os.path.join(DATA_DIR, name, "mito_seg.zarr"), "r")
    em  = zarr.open(os.path.join(DATA_DIR, name, "em.zarr"), "r")
    return feat_maps, seg, em, grid_shape


def extract_mito_embeddings(feat_maps, seg, grid_shape):
    """Mean-pool patch tokens within each mito mask -> one [768] vec per (z, mito_id)."""
    Z = feat_maps.shape[0]
    grid_h, grid_w = grid_shape
    H, W = seg.shape[1], seg.shape[2]

    embeddings = {}
    patch_counts = {}

    for z in range(Z):
        fm = feat_maps[z]  # [768, grid_h, grid_w]
        seg_grid = zoom(seg[z], (grid_h / H, grid_w / W), order=0)  # nearest-neighbor downsample

        for mito_id in np.unique(seg_grid[seg_grid > 0]):
            mask = seg_grid == mito_id
            key = (z, int(mito_id))
            embeddings[key] = fm[:, mask].mean(axis=1)  # [768, n_patches] -> [768]
            patch_counts[key] = int(mask.sum())

    return embeddings, patch_counts


def cosine_sim(query, matrix):
    """query: [768], matrix: [N, 768] -> [N]"""
    q = query / (np.linalg.norm(query) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m @ q


def get_mito_crop(em, seg, z, mito_id, pad=8):
    mask = seg[z] == mito_id
    rows, cols = np.where(mask)
    r0 = max(0, rows.min() - pad);  r1 = min(em.shape[1], rows.max() + pad + 1)
    c0 = max(0, cols.min() - pad);  c1 = min(em.shape[2], cols.max() + pad + 1)
    crop = em[z, r0:r1, c0:c1].astype(np.float32)
    lo, hi = np.percentile(crop, [1, 99])
    return np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1), mask[r0:r1, c0:c1]


def plot_retrieval(query_id, query_z, query_em, query_seg,
                   result_ids, result_zs, result_em, result_seg,
                   sims, title, out_path):
    K = len(result_ids)
    fig, axes = plt.subplots(1, K + 1, figsize=(3 * (K + 1), 3.5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    crop, mask = get_mito_crop(query_em, query_seg, query_z, query_id)
    axes[0].imshow(crop, cmap="gray")
    axes[0].contour(mask, colors="cyan", linewidths=0.8)
    axes[0].set_title("Query", fontsize=9)
    axes[0].axis("off")

    for i, (rid, rz, sim) in enumerate(zip(result_ids, result_zs, sims)):
        crop, mask = get_mito_crop(result_em, result_seg, rz, rid)
        axes[i + 1].imshow(crop, cmap="gray")
        axes[i + 1].contour(mask, colors="lime", linewidths=0.8)
        axes[i + 1].set_title(f"sim = {sim:.3f}", fontsize=9)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("loading datasets...")
    data = {}
    for name in DATASETS:
        fm, seg, em, grid_shape = load_dataset(name)
        data[name] = dict(feat_maps=fm, seg=seg, em=em, grid_shape=grid_shape)
        print(f"  {name}: feat_maps={fm.shape}  seg={seg.shape}")

    print("\nextracting mito embeddings...")
    embeddings, counts = {}, {}
    for name in DATASETS:
        d = data[name]
        emb, cnt = extract_mito_embeddings(d["feat_maps"], d["seg"], d["grid_shape"])
        embeddings[name] = emb
        counts[name] = cnt
        print(f"  {name}: {len(emb)} (z, mito_id) samples")

    query_key = max(counts["hela-2"], key=lambda k: counts["hela-2"][k])
    query_z, query_id = query_key
    print(f"\nquery: hela-2  mito_id={query_id}  z={query_z}  patches={counts['hela-2'][query_key]}")

    query_emb = embeddings["hela-2"][query_key]

    print("\nwithin-dataset retrieval (hela-2)...")
    other_keys = [k for k in embeddings["hela-2"] if k != query_key]
    other_vecs = np.stack([embeddings["hela-2"][k] for k in other_keys])
    sims_w = cosine_sim(query_emb, other_vecs)
    top_idx = np.argsort(sims_w)[::-1][:TOP_K]

    plot_retrieval(
        query_id, query_z, data["hela-2"]["em"], data["hela-2"]["seg"],
        result_ids=[other_keys[i][1] for i in top_idx],
        result_zs=[other_keys[i][0] for i in top_idx],
        result_em=data["hela-2"]["em"],
        result_seg=data["hela-2"]["seg"],
        sims=sims_w[top_idx],
        title=f"Within-dataset retrieval — hela-2  (query mito_id={query_id}, z={query_z})",
        out_path=os.path.join(OUTPUT_DIR, "task3_within_retrieval.png"),
    )

    print("cross-dataset retrieval (hela-2 → hela-3)...")
    cross_keys = list(embeddings["hela-3"].keys())
    cross_vecs = np.stack([embeddings["hela-3"][k] for k in cross_keys])
    sims_c = cosine_sim(query_emb, cross_vecs)
    top_idx_c = np.argsort(sims_c)[::-1][:TOP_K]

    plot_retrieval(
        query_id, query_z, data["hela-2"]["em"], data["hela-2"]["seg"],
        result_ids=[cross_keys[i][1] for i in top_idx_c],
        result_zs=[cross_keys[i][0] for i in top_idx_c],
        result_em=data["hela-3"]["em"],
        result_seg=data["hela-3"]["seg"],
        sims=sims_c[top_idx_c],
        title=f"Cross-dataset retrieval — hela-2 query → hela-3  (query mito_id={query_id}, z={query_z})",
        out_path=os.path.join(OUTPUT_DIR, "task3_cross_retrieval.png"),
    )

    print("\ndone.")


if __name__ == "__main__":
    main()
