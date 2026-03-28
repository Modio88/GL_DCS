
import numpy as np
import os, glob, time, rasterio
from sklearn.cluster import KMeans
import os



def clean_for_output(arr, nodata_value=0):
    arr = np.array(arr, dtype=float)  # 确保是浮点数
    arr[np.isnan(arr)] = nodata_value
    arr[np.isinf(arr)] = nodata_value
    return arr.astype(np.uint8)

def kmeans(image):
    rows, cols = image.shape

    # 将图像转换为一列向量
    image_reshaped = image.reshape(-1, 1)

    # ========== 2. K-means 聚类，k=2 ==========
    kmean = KMeans(n_clusters=2)
    kmean.fit(image_reshaped)

    # 获取每个像素的类别标签
    labels = kmean.labels_

    # ========== 3. 根据照片中心像元确定“中心类” ==========
    indices0 = np.where(labels == 0)[0]  # 一维
    std0 = indices0.std()
    indices1 = np.where(labels == 1)[0]  # 一维
    std1 = indices1.std()
    if std0<std1:
        center_label=0
    else:
        center_label=1
    # 将与中心像元同类的标记为1，其他为0
    binary_result = np.where(labels == center_label, 1, 0)

    # 恢复为原始图像大小
    binary_image = binary_result.reshape(rows, cols)
    return  binary_image
def md_kmeans(bands,mask):
    rows, cols = mask.shape
    data = None
    for key in sorted(bands.keys()):
        bands[key][np.isnan(bands[key])] = np.nanmin(bands[key])
        #归一化
        band_array = bands[key]
        # band_array = np.array(band_array, dtype=np.float64)
        # # band_array = np.around(band_array, 6)
        # band_array = (band_array - np.nanmin(band_array)) / (np.nanmax(band_array) - np.nanmin(band_array))  # 归一化

        # print(key, np.nanmax(band_array), np.nanmin(band_array))

        band_as_column = band_array.reshape(-1, 1)

        # if (key == 'vv') or (key == 'mir2') or(key == 'red') or (key == 'green'):
        #     band_as_column = band_as_column * 4
        #     band_as_column[band_as_column > 4] = 4

        data = band_as_column if data is None else np.concatenate([data, band_as_column], axis=1)
    kmean = KMeans(n_clusters=2)
    # kmean = AgglomerativeClustering(n_clusters=2)
    kmean.fit(data)

    # 获取每个像素的类别标签
    labels = kmean.labels_
    labels_img=labels.reshape(rows, cols)

    # ========== 3. 根据labels_id 对应vv 确定“water” ==========
    ids_label_0 = np.where(labels_img == 0)
    ids_label_1 = np.where(labels_img == 1)
    n=bands['nir']
    n0 = np.mean(n[ids_label_0])
    n1 = np.mean(n[ids_label_1])
    if n0>n1:
        label_water=1
    else:
        label_water=0
    # 将与中心像元同类的标记为1，其他为0
    binary_result = np.where(labels == label_water, 255, 0)

    # 恢复为原始图像大小
    binary_image = binary_result.reshape(rows, cols)
    return  binary_image

def md_clustering(img_data):

    NIR = img_data[0, :, :]
    ndwi = img_data[1, :, :]
    vv = img_data[2, :, :]
    bands={'nir':NIR,'ndwi':ndwi,'vv':vv}
    # bands = {'vv': vv}
    water_1 = ndwi.astype(float)
    mask = np.where(np.isnan(water_1),np.nan,1)
    water=md_kmeans(bands,mask)*mask
    return water
