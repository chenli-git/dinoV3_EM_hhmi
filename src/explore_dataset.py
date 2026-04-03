from fibsem_tools import read
import numpy as np
from PIL import Image
import os

os.makedirs("outputs", exist_ok=True)

datasets = {
    "hela-2": "s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.n5/em/fibsem-uint16/",
    "hela-3": "s3://janelia-cosem-datasets/jrc_hela-3/jrc_hela-3.n5/em/fibsem-uint16/"
}

for name, path in datasets.items():
    print(f"Dataset: {name}")

    group = read(
        path,
        storage_options={'anon': True}
    )

    # Print multiscale levels
    for level, arr in group.arrays():
        print(f"  {level}: shape={arr.shape}, dtype={arr.dtype}, chunks={arr.chunks}")

    # Print metadata
    print(f"  attrs: {dict(group.attrs)}")

    # Save one slice from s1 level
    s1 = group["s1"]
    mid_z = s1.shape[0] // 2
    print(f"  Saving s1 slice at z={mid_z} ...")
    slice_data = np.array(s1[mid_z])
    # Normalize to 0-255 for visualization
    #slice_norm = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    Image.fromarray(slice_data).save(f"outputs/{name}_s1_slice.tif")
    print(f"Saved outputs/{name}_s1_slice.tif")
