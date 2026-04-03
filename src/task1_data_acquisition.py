"""
Task 1: Data Acquisition
"""

import os
import json
import numpy as np
import zarr
import tifffile
from fibsem_tools import read

OUTPUT_DIR = "data"
NUM_SLICES = 5
CROP_X = 2048       # center-crop X to keep data manageable; full width is ~6000 px
SCALE = "s1"        # s0=4nm/px (too large), s1=8nm/px (good for DINO), s2=16nm/px (too coarse)

DATASETS = {
    "hela-2": {
        "em":   "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/",
        "mito": "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/labels/mito_seg/",
    },
    "hela-3": {
        "em":   "s3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.n5/em/fibsem-uint16/",
        "mito": "s3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.n5/labels/mito_seg/",
    },
}


def center_crop_slices(shape_zyx, num_slices):
    z, y, x = shape_zyx
    mid = z // 2
    z_idx = np.linspace(mid - 100, mid + 100, num_slices, dtype=int)
    cx = min(x, CROP_X)
    x_sl = slice(x // 2 - cx // 2, x // 2 - cx // 2 + cx)
    return z_idx, slice(None), x_sl   # keep full Y


def download(name, paths):
    out = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out, exist_ok=True)

    print(f"\n--- {name} ---")
    em_arr   = read(paths["em"],   storage_options={"anon": True})[SCALE]
    mito_arr = read(paths["mito"], storage_options={"anon": True})[SCALE]
    print(f"  EM   {SCALE}: {em_arr.shape}  {em_arr.dtype}")
    print(f"  Mito {SCALE}: {mito_arr.shape}  {mito_arr.dtype}")

    z_idx, y_sl, x_sl = center_crop_slices(em_arr.shape, NUM_SLICES)
    mito_z_idx = z_idx[z_idx < mito_arr.shape[0]]  

    print(f"  z-indices: {z_idx.tolist()}")

    print("  downloading EM slices...")
    em_data = np.stack([np.array(em_arr[z, y_sl, x_sl]) for z in z_idx])

    print("  downloading mito segmentation...")
    mito_data = np.stack([np.array(mito_arr[z, y_sl, x_sl]) for z in mito_z_idx])

    n_mito = len(np.unique(mito_data[mito_data > 0]))
    print(f"  shape: {em_data.shape} | {n_mito} unique mitochondria")

    # save as separate zarr arrays (chunks at root level, directly openable in Fiji)
    # keep raw uint16 — normalize later in Task 2
    for arr, fname in [(em_data, "em.zarr"), (mito_data, "mito_seg.zarr")]:
        z = zarr.open(os.path.join(out, fname), mode="w",
                      shape=arr.shape, chunks=(1, 512, 512), dtype=arr.dtype)
        z[:] = arr
        z.attrs["z_indices"] = z_idx.tolist()

    # save metadata
    meta = {
        "dataset":          name,
        "scale":            SCALE,
        "em_full_shape":    list(em_arr.shape),
        "mito_full_shape":  list(mito_arr.shape),
        "subset_shape":     list(em_data.shape),
        "z_indices":        z_idx.tolist(),
        "crop_x":           [x_sl.start, x_sl.stop],
        "num_mitochondria": n_mito,
    }
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # save multi-page tiff stacks (uint16) — open directly in Fiji with File -> Open
    tifffile.imwrite(os.path.join(out, "em_stack.tif"),   em_data)
    tifffile.imwrite(os.path.join(out, "mito_stack.tif"), mito_data)

    print(f"  saved to {out}/")
    return meta


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {}
    for name, paths in DATASETS.items():
        summary[name] = download(name, paths)
    with open(os.path.join(OUTPUT_DIR, "datasets_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\ndone.")


if __name__ == "__main__":
    main()
