import os
from boundary import boundary_water
tif_dir = r"E:/study/glof/sentinel2/images512/"    # 存放原始GeoTIFF的文件夹
lake_labels_path = r"E:/study/glof/sentinel2/labels_yolo/"      # 存放YOLO结果TXT的文件夹
water_dir = r"E:/study/glof/sentinel2/water/water_combine/"  # 输出裁剪结果
os.makedirs(water_dir, exist_ok=True)
boundary_water(tif_dir, lake_labels_path, water_dir)