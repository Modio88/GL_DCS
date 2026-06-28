import os
import numpy as np
import rasterio
from rasterio.transform import Affine

# =================== 配置路径 ===================
tif_folder = r"E:\study\glof\test\images_512"    # 存放原始GeoTIFF的文件夹
txt_folder = r"E:\study\glof2\test\labels_yolo"      # 存放YOLO结果TXT的文件夹
output_folder1 = r"E:/study/glof2/test/labels_shp/tif/"  # 输出裁剪结果
# output_folder2 = r"E:/study/glof2/sentinel2/labels_shp/"  # 输出裁剪结果
os.makedirs(output_folder1, exist_ok=True)

# =================== 工具函数 ===================
def yolo_to_pixel(bbox, img_width, img_height):
    """
    将YOLO归一化坐标转换为像素坐标
    bbox = [center_x, center_y, w, h]
    返回 [col_min, row_min, col_max, row_max]
    """
    expand = 0
    cx, cy, w, h = bbox

    box_width = w * img_width
    box_height = h * img_height

    col_min = int(cx * img_width - box_width / 2 - expand)
    row_min = int(cy * img_height - box_height / 2 - expand)
    col_max = int(cx * img_width + box_width / 2 + expand)
    row_max = int(cy * img_height + box_height / 2 + expand)

    return col_min, row_min, col_max, row_max

# =================== 主批处理流程 ===================
def labels_to_tif(tif_folder, txt_folder,output_folder):
    total_tif = 0
    total_txt_found = 0
    total_crops = 0

    # 遍历所有tif文件
    for tif_file in os.listdir(tif_folder):
        if not tif_file.lower().endswith(".tif"):
            continue

        total_tif += 1
        base_name = os.path.splitext(tif_file)[0]

        # 对应的txt文件路径
        txt_path = os.path.join(txt_folder, base_name + ".txt")

        if not os.path.exists(txt_path):
            print(f"[跳过] 未找到对应txt文件: {base_name}.txt")
            continue

        total_txt_found += 1

        tif_path = os.path.join(tif_folder, tif_file)

        # ========== 读取tif文件 ==========
        with rasterio.open(tif_path) as src:
            img_data = src.read()  # shape: (bands, height, width)
            ref_crs = src.crs
            ref_transform = src.transform
            ref_width = src.width
            ref_height = src.height
            blue = img_data[0, :, :]
            img_array = (blue * 0 +255).astype(np.uint8)

        # ========== 读取YOLO结果txt ==========
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # ========== 逐行裁剪 ==========
        crop_index = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            class_id = parts[0]
            bbox = list(map(float, parts[1:5]))

            # 1. YOLO -> 像素坐标
            col_min, row_min, col_max, row_max = yolo_to_pixel(bbox, ref_width, ref_height)

            # 2. 边界检查，防止超出影像范围
            col_min = max(0, col_min)
            row_min = max(0, row_min)
            col_max = min(ref_width, col_max)
            row_max = min(ref_height, row_max)

            if col_min >= col_max or row_min >= row_max:
                print(f"[跳过] 无效边界框: {line}")
                continue

            # 3. 裁剪
            crop_array = img_array[row_min:row_max, col_min:col_max]

            # 4. 计算新的transform
            new_transform = ref_transform * Affine.translation(col_min, row_min)

            # 5. 输出路径
            output_name = f"{base_name}_class{class_id}_crop_{i+1}.tif"
            output_path = os.path.join(output_folder, output_name)

            # 6. 保存裁剪结果
            profile = {
                'driver': 'GTiff',
                'dtype': crop_array.dtype,
                'count': 1,
                'width': crop_array.shape[1],
                'height': crop_array.shape[0],
                'crs': ref_crs,
                'transform': new_transform
            }

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(crop_array,1)

            crop_index += 1
            total_crops += 1
    #         print(f"[完成] 保存裁剪结果: {output_path}")
    #
    #     print(f"[完成] {tif_file} -> {crop_index} 个裁剪结果")
    #
    # # =================== 总结 ===================
    # print("\n================= 处理完成 =================")
    # print(f"共找到 {total_tif} 个tif文件")
    # print(f"其中 {total_txt_found} 个tif文件有对应的txt文件")
    # print(f"总共裁剪输出 {total_crops} 个GeoTIFF文件")


