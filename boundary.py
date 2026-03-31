import os
import torch
import numpy as np
import rasterio
from rasterio.transform import Affine

from data_process.mutiple_dimension_kmeans import md_clustering
from sam_package.sam2.build_sam import build_sam2
from sam_package.sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
#目的是在512尺度上整体归一化

# =====================
# 基本配置


def segment(img,box,predictor):
    img = img.astype(np.float32)

    # 自动归一化（更稳）
    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    # 转为 (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # 若是多光谱，取前三波段（SAM 需要 3 通道）
    if img.shape[2] > 3:
        img = img[:, :, :3]

    # 若是单波段，复制成 3 通道
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # -------- SAM2 推理 --------
    predictor.set_image(img)

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=False,

    )

    # masks: (1, H, W)
    mask = masks[0]
    return (mask > 0).astype(np.uint8) * 255


# =================== 工具函数 ===================
def yolo_to_pixel(bbox, img_width, img_height):
    """
    将YOLO归一化坐标转换为像素坐标
    bbox = [center_x, center_y, w, h]
    返回 [col_min, row_min, col_max, row_max]
    """
    expand = 20
    cx, cy, w, h = bbox

    box_width = w * img_width
    box_height = h * img_height

    col_min = int(cx * img_width - box_width / 2 - expand)
    row_min = int(cy * img_height - box_height / 2 - expand)
    col_max = int(cx * img_width + box_width / 2 + expand)
    row_max = int(cy * img_height + box_height / 2 + expand)

    return col_min, row_min, col_max, row_max
def ndwi(green, NIR):
    AB = np.array(green + NIR, dtype=np.float32)
    # AC = np.array(green - NIR, dtype=np.float32)
    AC = np.array(NIR -green, dtype=np.float32)
    nodata = np.full((green.shape[0], green.shape[1]), -2, dtype=np.float32)
    ndwi = np.true_divide(AC, AB)
    ndwi[ndwi == -2.0] = None
    return ndwi
def is_dark(rgb, threshold=80):
    """
    判断影像是否偏暗
    rgb: (H, W, 3), uint8
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # luminance = r + g + b
    return luminance.mean() < threshold


def clahe_on_rgb(rgb,clip_limit,tile_grid_size):
    """
    只调整亮度（Lab 的 L 通道）
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l2 = clahe.apply(l)

    lab2 = cv2.merge((l2, a, b))
    rgb_out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return rgb_out

# =================== 主批处理流程 ===================
# =====================


def boundary_water(tif_folder,txt_folder,output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # =====================
    # 构建 SAM2 模型
    # =====================
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
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
            green = img_data[1, :, :]
            red = img_data[2, :, :]
            NIR = img_data[3, :, :]
            # swir1 = img_data[4, :, :]
            vv = img_data[4, :, :]
            ndwi_index = ndwi(green, NIR)  # 1 ndwi
            bands = {'a': NIR, 'b': ndwi_index, 'c': vv}   #修改参与计算的波段
            # bands = {'vv': vv}
            mask0 = np.where(np.isnan(ndwi_index.astype(float)), np.nan, 1)
            outimg_list = []
            for key in sorted(bands.keys()):
                bands[key][np.isnan(bands[key])] = np.nanmin(bands[key])
                band_array = bands[key].astype(np.float64)
                band_array =255* (band_array - np.nanmin(band_array)) / (
                            np.nanmax(band_array) - np.nanmin(band_array))* mask0
                outimg_list.append(band_array)
            img_array = np.array(outimg_list)
            #直方图均值化
            dark_threshold = 80  # 偏暗判定阈值（0–255）
            clip_limit = 1.5  # CLAHE 对比度限制
            tile_grid_size = (4, 4)  # CLAHE 分块大小
            rgb = np.stack([
                img_array[0],
                img_array[1],
                img_array[2]
            ], axis=-1)
            if is_dark(rgb, dark_threshold):
                rgb_enhanced = clahe_on_rgb(rgb, clip_limit, tile_grid_size)

                # 写回前三个波段
                img_array[0] = rgb_enhanced[..., 0]
                img_array[1] = rgb_enhanced[..., 1]
                img_array[2] = rgb_enhanced[..., 2]
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
            padd=20
            padding1 = padd
            padding2 = padd
            padding3 = padd
            padding4 = padd
            if col_min<0:
                padding1=padd+col_min
            if row_min<0:
                padding2=padd+row_min
            if col_max>ref_width:
                padding3=padd-(col_max-ref_width)
            if row_max>ref_height:
                padding4=padd-(row_max-ref_height)
            col_min = max(0, col_min)
            row_min = max(0, row_min)
            col_max = min(ref_width, col_max)
            row_max = min(ref_height, row_max)

            if col_min >= col_max or row_min >= row_max:
                print(f"[跳过] 无效边界框: {line}")
                continue

            # 3. 裁剪
            crop_array = img_array[:, row_min:row_max, col_min:col_max]
            c, hh, ww = crop_array.shape
            box = np.array([[padding1, padding2, ww - padding3, hh - padding4]])
            #归一化
            mask=segment(crop_array,box,predictor)
            # sam未分离water
            if not np.any(mask == 255):
                col_min = max(0, col_min + 15)
                row_min = max(0, row_min + 15)
                col_max = min(ref_width, col_max - 15)
                row_max = min(ref_height, row_max - 15)
                # 3. 裁剪
                crop_array = img_array[:, row_min:row_max, col_min:col_max]
                mask = md_clustering(crop_array)

            # 4. 计算新的transform
            new_transform = ref_transform * Affine.translation(col_min, row_min)

            # 5. 输出路径
            output_name = f"{base_name}_class{class_id}_crop_{i+1}.tif"
            # output_img = os.path.join(output_folder, output_name)
            output_mask = os.path.join(output_dir, output_name)
            # 6. 保存裁剪结果
            profile = {
                'driver': 'GTiff',
                "dtype": rasterio.uint8,
                'count': 1,
                'width': crop_array.shape[2],
                'height': crop_array.shape[1],
                'crs': ref_crs,
                'transform': new_transform
            }
            # 储存影像
            # with rasterio.open(output_img, 'w', **profile1) as dst:
            #     dst.write(crop_array,1)


            with rasterio.open(output_mask, 'w', **profile) as dst:
                dst.write(mask,1)

            crop_index += 1
            total_crops += 1

#cd /mnt/e/study/glof/sam/sam2
#python predict2.py