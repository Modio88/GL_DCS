import os
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import box
from rasterio.features import geometry_mask


def normalize(array):
    """2%-98%拉伸到0-255"""
    array = array.astype(float)
    p2, p98 = np.nanpercentile(array, (1, 99))
    array = np.clip(array, p2, p98)
    return ((array - p2) / (p98 - p2) * 255).astype(np.uint8)
def tif_to_rgb(tif_path):
    with rasterio.open(tif_path) as src:
        tif_data = src.read()
        nodata = src.nodata

        # 如果实际顺序不同，请调整索引
        band1 = tif_data[0].astype(float)
        band2 = tif_data[1].astype(float)
        band3 = tif_data[2].astype(float)

        # 替换NoData为nan
        if nodata is not None:
            band1[band1 == nodata] = np.nan
            band2[band2 == nodata] = np.nan
            band3[band3 == nodata] = np.nan

        # 归一化到0-255
        r = normalize(band3)
        g = normalize(band2)
        b = normalize(band1)

        rgb = np.dstack((r, g, b))
        return rgb
# rgb=tif_to_rgb('E:\study\glof\glacial_lake_system\images\S_test1_2023.tif')
# print(rgb.shape)