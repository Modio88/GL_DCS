# -*- coding: utf-8 -*-
"""
将 GeoTIFF 影像切成 512x512 小瓦片，保留坐标系/地理参考；
若波段1全为 NoData，则不输出该瓦片。
依赖: rasterio, numpy, tqdm
安装: pip install rasterio numpy tqdm
"""

import os
from math import ceil
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from tqdm import tqdm

def tile_geotiff(
    input_tif: str,
    out_dir: str,
    fname: str,
    tile_size: int = 512,
    overlap: int = 0,
    pad: bool = False
):
    """
    将 GeoTIFF 影像切成 tile_size x tile_size 的瓦片。
    若波段1全为空值（NoData），则不输出该瓦片。
    """

    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(input_tif) as src:
        width, height = src.width, src.height
        count, dtype = src.count, src.dtypes[0]
        crs = src.crs
        nodata = src.nodata
        if nodata is None:
            nodata = 0 if np.issubdtype(np.dtype(dtype), np.integer) else np.nan

        # 步长（考虑重叠）
        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError("overlap 必须小于 tile_size。")

        # 计算瓦片数量
        nx = ceil((width  - overlap) / stride) if pad else max(1, (width  - overlap) // stride)
        ny = ceil((height - overlap) / stride) if pad else max(1, (height - overlap) // stride)

        total_tiles = nx * ny
        pbar = tqdm(total=total_tiles, desc=f"Tiling: {os.path.basename(input_tif)}")

        valid_tile_count = 0

        for j in range(ny):
            for i in range(nx):
                col_off = i * stride
                row_off = j * stride
                w = min(tile_size, width  - col_off)
                h = min(tile_size, height - row_off)

                if w <= 0 or h <= 0:
                    pbar.update(1)
                    continue

                window = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                data = src.read(window=window)

                # 若 pad=True，边缘补齐
                if pad and (w < tile_size or h < tile_size):
                    fill_val = nodata if not (isinstance(nodata, float) and np.isnan(nodata)) else np.nan
                    padded = np.full((count, tile_size, tile_size), fill_val, dtype=dtype)
                    padded[:, :h, :w] = data
                    data = padded
                    out_h, out_w = tile_size, tile_size
                else:
                    out_h, out_w = h, w

                # 检查波段1是否全为 nodata
                band1 = data[0]
                if np.isnan(nodata):
                    all_nodata = np.isnan(band1).all()
                else:
                    all_nodata = np.all(band1 == nodata)

                # 若全为空值，则跳过
                if all_nodata:
                    pbar.update(1)
                    continue

                # 计算 transform
                tile_transform: Affine = rasterio.windows.transform(window, src.transform)

                # 更新 profile
                profile = src.profile.copy()
                profile.update({
                    "height": out_h,
                    "width": out_w,
                    "transform": tile_transform,
                    "nodata": nodata
                })

                out_name = fname+f"_r{j:02d}_c{i:02d}.tif"
                out_path = os.path.join(out_dir, out_name)

                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)

                valid_tile_count += 1
                pbar.update(1)

        pbar.close()
        # print(f"完成：共输出 {valid_tile_count} 个有效瓦片到 {out_dir}")

if __name__ == "__main__":
    in_tif = "E:/study/glof/test/images/"
    out_dir = "E:/study/glof/test/images_512/"
    for tif_file in os.listdir(in_tif):
        if not tif_file.lower().endswith(".tif"):
            continue
        tif_path = os.path.join(in_tif, tif_file)
        fname=tif_file[:10]
        tile_geotiff(tif_path, out_dir, fname,tile_size=512, overlap=16, pad=True) #overlap=16
