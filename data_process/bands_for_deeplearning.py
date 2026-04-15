# 将波段归一化输出成tif
import cv2
import numpy as np
import shutil
import rasterio.plot
import geopandas as gpd
import os, glob, time, rasterio
from osgeo import gdal, osr, ogr, gdalconst
import os
from scipy.signal import convolve2d
import math

def ndwi(green, NIR):
    AB = np.array(green + NIR, dtype=np.float32)
    AC = np.array(green - NIR, dtype=np.float32)
    nodata = np.full((green.shape[0], green.shape[1]), -2, dtype=np.float32)
    ndwi = np.true_divide(AC, AB)
    # print(np.nanmin(ndwi), np.nanmax(ndwi))
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
## 读图像文件
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    # im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
    im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset  # 关闭对象dataset，释放内存
    # return im_width, im_height, im_proj, im_geotrans, im_data,im_bands
    return im_proj, im_geotrans, im_data, im_width, im_height
def write_Tiff(img_arr, geomatrix, projection, path):
    #     img_bands, img_height, img_width = img_arr.shape
    if 'int8' in img_arr.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_arr.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_arr.shape) == 3:
        img_bands, img_height, img_width = img_arr.shape
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(img_width), int(img_height), int(img_bands), datatype)
        #     print(path, int(img_width), int(img_height), int(img_bands), datatype)
        if (dataset != None) and (geomatrix != '') and (projection != ''):
            dataset.SetGeoTransform(geomatrix)  # 写入仿射变换参数
            dataset.SetProjection(projection)  # 写入投影
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img_arr[i])
        del dataset

    elif len(img_arr.shape) == 2:
        # img_arr = np.array([img_arr])
        img_height, img_width = img_arr.shape
        img_bands = 1
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(img_width), int(img_height), int(img_bands), datatype)
        #     print(path, int(img_width), int(img_height), int(img_bands), datatype)
        if (dataset != None):
            dataset.SetGeoTransform(geomatrix)  # 写入仿射变换参数
            dataset.SetProjection(projection)  # 写入投影
        dataset.GetRasterBand(1).WriteArray(img_arr)
        del dataset


def to_tiff(input_path,out_path):
    AA = glob.glob(os.path.join(input_path, '*.tif'))

    for i in range(0, len(AA)):
        new_stack = AA[i]
        proj, geotrans, img_data, row, column = read_img(new_stack)
        AD = os.path.basename(AA[i])  # 读取文件名
        # print(img_data)
        blue = img_data[0, :, :]
        green = img_data[1, :, :]
        red = img_data[2, :, :]
        NIR = img_data[3, :, :]
        vv=img_data[4,:,:]
        dem=img_data[5,:,:]
        ndwi_index = ndwi(green, NIR)  # 1 ndwi

        # 计算坡度
        pixel_size = geotrans[1]  # x方向像元大小
        if abs(geotrans[5]) != pixel_size:  # 检查y方向像元大小
            print(f"警告：x和y方向像元大小不同 ({pixel_size} vs {abs(geotrans[5])})")
            pixel_size = (pixel_size + abs(geotrans[5])) / 2  # 取平均值
            # 创建9×9卷积核
        w = np.arange(-4, 5)  # [-4, -3, ..., 4]
        kx = np.tile(w, (9, 1))
        ky = np.tile(w.reshape(9, 1), (1, 9))

        # 归一化（使权重总和为0，但尺度一致）
        kx = kx / (np.sum(np.abs(kx)))
        ky = ky / (np.sum(np.abs(ky)))

        # 计算dz/dx, dz/dy
        dz_dx = convolve2d(dem, kx, mode='same', boundary='symm') / pixel_size
        dz_dy = convolve2d(dem, ky, mode='same', boundary='symm') / pixel_size

        # 计算坡度（度）
        slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

        # 掩膜无效值
        mask = np.where(np.isnan(ndwi_index.astype(float)), np.nan, 1)
        # bands={'b1':red,'b2':green,'b3':blue,'b4':NIR,'b5':ndwi_index,'b6':vv,'b7':dem,'b8':slope}
        bands = {'b1': red, 'b2': green,'b3':blue,'b4':NIR,'b5':ndwi_index,'b6':vv,'b7':dem,'b8':slope}
        outimg_list = []
        for key in sorted(bands.keys()):
            bands[key] = bands[key] * mask
            bands[key][np.isnan(bands[key])] = np.nanmin(bands[key])
            band_array = bands[key].astype(np.float32)
            p2, p98 = np.nanpercentile(band_array, (1, 99))
            band_array = np.clip(band_array, p2, p98)
            band_array =(255* (band_array - np.nanmin(band_array)) / (np.nanmax(band_array) - np.nanmin(band_array))).astype(np.uint8)
            outimg_list.append(band_array)
        outimg = np.array(outimg_list)
        #对于偏暗的影像进行自适应直方图均衡化
        if len(bands)>2:
            dark_threshold = 80  # 偏暗判定阈值（0–255）
            clip_limit = 1.5  # CLAHE 对比度限制
            tile_grid_size = (4, 4)  # CLAHE 分块大小
            rgb = np.stack([
                outimg[0],
                outimg[1],
                outimg[2]
            ], axis=-1)
            if is_dark(rgb, dark_threshold):

                rgb_enhanced = clahe_on_rgb(rgb,clip_limit,tile_grid_size)

                # 写回前三个波段
                outimg[0] = rgb_enhanced[..., 0]
                outimg[1] = rgb_enhanced[..., 1]
                outimg[2] = rgb_enhanced[..., 2]

        out_AE = os.path.join(out_path,AD[:-4] + '.tiff')
        cv2.imwritemulti(out_AE, outimg)


