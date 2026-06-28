
import os

from boundary import boundary_water
from boundary_new import boundary_watern
from data_process.clipe_image import tile_geotiff
from data_process.bands_for_deeplearning import to_tiff
from data_process.labels_shp_clipeimage import shp_clipe_tif
from data_process.labelsyolo_toshp import labels_to_tif
from data_process.tif_to_shp_new import tif_to_shpn
from detect import detect_gl
from data_process.delet_dir import delete_folder_if_exists
from data_process.join_table import join_table_xlsx
from data_process.lake_clipe_dimage import lake_clipe
from predict_cls import classify_gl
from data_process.tif_to_shp import tif_to_shp
from data_process.to_rgb import tif_to_rgb
def gl_detect(in_tif,band):
    for tif_file in os.listdir(in_tif):
        if not tif_file.lower().endswith(".tif"):
            continue
        #初始化
        rgb=[]
        final_gdf=[]
        tif_path = os.path.join(in_tif, tif_file)
        rgb = tif_to_rgb(tif_path)
        fname = tif_file.rsplit('.', 1)[0]
        out_p = os.path.join(in_tif, fname)
        # 1裁剪影像 并将512影像转化为tiff
        name512 = fname + 'cropped'
        tif_dir = os.path.join(out_p, name512)  # 裁剪输出
        os.makedirs(tif_dir, exist_ok=True)
        name_tiff = fname + 'tiff'
        tiff_dir = os.path.join(out_p, name_tiff)  # 转化输出
        os.makedirs(tiff_dir, exist_ok=True)

        tile_geotiff(tif_path, tif_dir, tiff_dir, fname, 512, overlap=16, pad=True)  # overlap=16
        print("1裁剪&转化tiff完成")

        # 2冰湖检测
        name_txt = fname + 'detect'
        txt_dir = os.path.join(out_p, name_txt)  # 转化输出
        delete_folder_if_exists(txt_dir)
        os.makedirs(txt_dir, exist_ok=True)
        detect_gl(tiff_dir, txt_dir, band)
        print("3冰湖检测完成")
        if os.path.exists(txt_dir) and len(os.listdir(txt_dir)) == 0:
            print(f"⚠️  检测结果文件夹为空，跳过当前：{fname}，处理下一张影像")
            continue  # 直接进入下一个循环

        lake_labels_path = os.path.join(txt_dir, 'labels')

        # 6边界提取
        name_water = fname + 'water'
        water_dir = os.path.join(out_p, name_water)
        os.makedirs(water_dir, exist_ok=True)
        boundary_water(tif_dir, lake_labels_path, water_dir)
        print("6冰湖边界提取完成")
        # 7转化为shp
        shp_path = os.path.join(out_p, fname + '.shp')
        final_gdf=tif_to_shp(water_dir)
        final_gdf.to_file(shp_path)
        print("7冰湖矢量地图完成")
        #删除文件件
        delete_folder_if_exists(tif_dir)
        delete_folder_if_exists(tiff_dir)


    return rgb, final_gdf
def gl_cls(in_tif):
    for tif_file in os.listdir(in_tif):
        if not tif_file.lower().endswith(".tif"):
            continue
        # 初始化
        rgb = []
        final_gdf = []
        tif_path = os.path.join(in_tif, tif_file)
        rgb = tif_to_rgb(tif_path)
        fname = tif_file.rsplit('.', 1)[0]
        out_p = os.path.join(in_tif, fname)
        # 1裁剪影像 并将512影像转化为tiff
        name512 = fname + 'cropped'
        tif_dir = os.path.join(out_p, name512)  # 裁剪输出
        os.makedirs(tif_dir, exist_ok=True)
        name_tiff = fname + 'tiff'
        tiff_dir = os.path.join(out_p, name_tiff)  # 转化输出
        os.makedirs(tiff_dir, exist_ok=True)

        tile_geotiff(tif_path, tif_dir, tiff_dir, fname, 512, overlap=16, pad=True)  # overlap=16
        print("1裁剪&转化tiff完成")

        # 2冰湖检测
        name_txt = fname + 'detect'
        txt_dir = os.path.join(out_p, name_txt)  # 转化输出
        delete_folder_if_exists(txt_dir)
        os.makedirs(txt_dir, exist_ok=True)
        detect_gl(tiff_dir, txt_dir, band=8)
        print("3冰湖检测完成")
        if os.path.exists(txt_dir) and len(os.listdir(txt_dir)) == 0:
            print(f"⚠️  检测结果文件夹为空，跳过当前：{fname}，处理下一张影像")
            continue  # 直接进入下一个循环

        lake_labels_path = os.path.join(txt_dir, 'labels')
        # 3生成冰湖边界框矢量
        labels_tif = os.path.join(out_p, fname + 'labels_tif')  # 裁剪输出
        os.makedirs(labels_tif, exist_ok=True)
        labels_to_tif(tif_dir, lake_labels_path, labels_tif)
        labels_shp = os.path.join(labels_tif, fname + 'label.shp')
        labels_gdf=tif_to_shpn(labels_tif)
        labels_gdf.to_file(labels_shp)
        # 4labels_shp裁剪原图并生成lake_tiff
        lake_tif = os.path.join(out_p, fname + 'lake_tif')
        os.makedirs(lake_tif, exist_ok=True)
        cls_tiff = os.path.join(out_p, fname + 'cls_tiff')
        os.makedirs(cls_tiff, exist_ok=True)
        shp_clipe_tif(tif_path, labels_shp, lake_tif, cls_tiff, "cls")
        #5冰湖分类
        cls_file = os.path.join(out_p, fname+'cls')
        classify_gl(cls_tiff,cls_file)
        print("5冰湖分类完成")
        #6边界提取
        name_water=fname+'water'
        water_dir=os.path.join(out_p,name_water)
        os.makedirs(water_dir, exist_ok=True)
        boundary_watern(lake_tif, water_dir)
        print("6冰湖边界提取完成")
        #7转化为shp
        shp_path=os.path.join(out_p, fname+'.shp')
        final_gdf = tif_to_shp(water_dir)
        final_gdf.to_file(shp_path)
        print("7冰湖矢量地图完成")
        #8属性连接
        xlsx_path=os.path.join(cls_file,'cls_gl.xlsx')
        output_shp=os.path.join(in_tif, fname+'cls.shp')
        gdf_joined=join_table_xlsx(shp_path, xlsx_path)
        gdf_joined.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")
        print("8属性连接完成")
        # 删除文件件
        # delete_folder_if_exists(tif_dir)
        # delete_folder_if_exists(tiff_dir)
        # delete_folder_if_exists(labels_tif)
        # delete_folder_if_exists(lake_tif)
        # delete_folder_if_exists(cls_tiff)
        # delete_folder_if_exists(water_dir)

    return rgb,gdf_joined


