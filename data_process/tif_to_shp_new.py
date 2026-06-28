# 只合并来自不同 tif 重叠的多边形（自动转米制坐标系计算面积）

import os
import pandas as pd
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import networkx as nx
from pyproj import CRS

input_folder = r"E:\study\glof2\sentinel2\labels_shp\train11"
output_shp = r"E:\study\glof2\sentinel2\labels_shp\train11.shp"


def tif_to_shpn(input_folder):

    all_geoms = []
    all_names = []

    crs = None

    # =========================
    # 1 读取 tif 并转 polygon
    # =========================
    for file in os.listdir(input_folder):

        if file.lower().endswith(".tif"):

            path = os.path.join(input_folder, file)

            name = os.path.splitext(file)[0]

            with rasterio.open(path) as src:

                image = src.read(1)

                transform = src.transform

                # 获取 CRS
                src_crs = src.crs

                # 保存第一个有效 CRS
                if crs is None:
                    crs = src_crs

                for geom, value in shapes(image, transform=transform):

                    if value > 0:

                        g = shape(geom)

                        all_geoms.append(g)

                        all_names.append(name)

            print("完成:", file)

    # =========================
    # 2 构建 GeoDataFrame
    # =========================

    if crs is None:
        raise ValueError("所有 tif 都没有 CRS 坐标系信息")

    gdf = gpd.GeoDataFrame(
        {
            "filename": all_names
        },
        geometry=all_geoms,
        crs=crs
    )

    # =========================
    # 3 自动转换为米制坐标系
    # =========================

    # 如果是经纬度坐标系（单位=度）
    if CRS(gdf.crs).is_geographic:

        print("检测到经纬度坐标系，自动转换为 UTM 米制坐标系")

        # 自动估算最合适 UTM
        metric_crs = gdf.estimate_utm_crs()

        gdf_metric = gdf.to_crs(metric_crs)

    else:

        print("检测到投影坐标系，直接使用")

        metric_crs = gdf.crs

        gdf_metric = gdf.copy()

    # 面积（平方米）
    gdf["pixel_area"] = gdf_metric.geometry.area

    print("面积单位: 平方米 m²")

    # =========================
    # 4 构建空间索引
    # =========================

    sindex = gdf.sindex

    # 图结构
    G = nx.Graph()

    # 添加节点
    for idx in gdf.index:
        G.add_node(idx)

    # =========================
    # 5 查找不同 tif 之间的相交关系
    # =========================

    for idx1, row1 in gdf.iterrows():

        geom1 = row1.geometry

        # 空间索引查询
        possible_matches = list(
            sindex.intersection(geom1.bounds)
        )

        for idx2 in possible_matches:

            # 避免自己比较
            if idx1 >= idx2:
                continue

            row2 = gdf.loc[idx2]

            # 仅不同 tif
            name1 = row1.filename.split("_class")[0]
            name2 = row2.filename.split("_class")[0]

            # 同源 tif 跳过
            if name1 == name2:
                continue

            geom2 = row2.geometry

            # 真正相交
            if geom1.intersects(geom2):

                # 建立连接
                G.add_edge(idx1, idx2)

    # =========================
    # 6 连通域合并
    # =========================

    merged_geoms = []

    components = list(nx.connected_components(G))

    print("连通区域数量:", len(components))

    for comp in components:

        comp = list(comp)

        subset = gdf.loc[comp]

        # 合并 geometry
        merged_geom = unary_union(subset.geometry)

        # =========================
        # 面积转米制计算
        # =========================

        merged_geom_metric = gpd.GeoSeries(
            [merged_geom],
            crs=gdf.crs
        ).to_crs(metric_crs)

        merged_area = merged_geom_metric.area.iloc[0]

        # 面积过滤（可选）
        if merged_area < 1000:
            continue

        # 找最大原始面
        max_idx = subset["pixel_area"].idxmax()

        final_name = subset.loc[max_idx, "filename"]

        merged_geoms.append({
            "filename": final_name,
            "area_m2": merged_area,
            "geometry": merged_geom
        })

    # =========================
    # 7 输出
    # =========================

    final_gdf = gpd.GeoDataFrame(
        merged_geoms,
        crs=gdf.crs
    )

    return final_gdf


# # =========================
# # 主程序
# # =========================
#
# final_gdf = tif_to_shpn(input_folder)
#
# # 保存 shp
# final_gdf.to_file(output_shp, encoding="utf-8")
#
# print("完成")
# print("最终要素数:", len(final_gdf))
# print("输出:", output_shp)