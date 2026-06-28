import os

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from rasterio.windows import Window
import numpy as np
from data_process.bands_for_deeplearning import to_tiff

# =========================
# 输入参数
# =========================
tif_path = r"E:\study\glof\sentinel2\images\S_train11_.tif"
shp_path = r"E:\study\glof2\sentinel2\labels_shp\train11.shp"
out_dir = r"E:\study\glof2\sentinel2\cls_tif"
def shp_clipe_tif(tif_path,shp_path,out_dir,tiff_dir,task):
    # shp中的字段名
    # shp中的字段名
    filename_field = "filename"

    # 向外扩展像元数
    expand_pixels = 80

    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # 读取 shp
    # =========================
    gdf = gpd.read_file(shp_path)

    # =========================
    # 打开 tif
    # =========================
    with rasterio.open(tif_path) as src:

        transform = src.transform

        # 像元大小
        pixel_width = transform.a
        pixel_height = abs(transform.e)

        # nodata 值
        nodata_value = src.nodata

        # 如果原图没有 nodata
        if nodata_value is None:

            # float 类型可以用 nan
            if np.issubdtype(src.dtypes[0], np.floating):
                nodata_value = np.nan
            else:
                # 整型给一个常见 nodata
                nodata_value = -9999

        for idx, row in gdf.iterrows():

            geom = row.geometry

            # =========================
            # 获取外接矩形
            # =========================
            minx, miny, maxx, maxy = geom.bounds

            # =========================
            # 按像元扩展
            # =========================
            minx_expand = minx - expand_pixels * pixel_width
            maxx_expand = maxx + expand_pixels * pixel_width

            miny_expand = miny - expand_pixels * pixel_height
            maxy_expand = maxy + expand_pixels * pixel_height

            # =========================
            # 转换为 window
            # =========================
            window = from_bounds(
                minx_expand,
                miny_expand,
                maxx_expand,
                maxy_expand,
                transform=src.transform
            )

            # window整数化
            window = window.round_offsets().round_lengths()

            # =========================
            # 读取数据
            # boundless=True:
            # 允许越界
            # fill_value:
            # 越界区域填充值
            # =========================
            clipped = src.read(
                window=window,
                boundless=True,
                fill_value=nodata_value
            )

            # =========================
            # 更新 transform
            # =========================
            out_transform = rasterio.windows.transform(
                window,
                src.transform
            )

            # =========================
            # 输出文件名
            # =========================
            out_name = str(row[filename_field]) + ".tif"
            out_path = os.path.join(out_dir, out_name)

            # =========================
            # 更新 metadata
            # =========================
            out_meta = src.meta.copy()

            out_meta.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": out_transform,
                "nodata": nodata_value
            })

            # float + nan 时建议使用 float32
            if np.isnan(nodata_value):
                out_meta.update({
                    "dtype": "float32"
                })
                clipped = clipped.astype(np.float32)

            # =========================
            # 保存 tif
            # =========================
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(clipped)

            print(f"Saved: {out_path}")

            # =========================
            # 转 tiff
            # =========================
            if task == "cls" and tiff_dir is not None:
                os.makedirs(tiff_dir, exist_ok=True)

                tiff_name = str(row[filename_field]) + ".tiff"
                tiff_path = os.path.join(tiff_dir, tiff_name)

                # 你自己的函数
                to_tiff(clipped, tiff_path)

    #         print(f"输出完成: {out_path}")
    #
    # print("全部裁剪完成")