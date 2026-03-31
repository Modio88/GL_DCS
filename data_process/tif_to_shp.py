import os
import pandas as pd
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union


def tif_to_shp(input_folder):

    all_geoms = []
    all_names = []
    all_areas = []

    # 1 读取tif并转polygon
    for file in os.listdir(input_folder):

        if file.lower().endswith(".tif"):

            path = os.path.join(input_folder, file)
            name = os.path.splitext(file)[0]

            with rasterio.open(path) as src:

                image = src.read(1)
                transform = src.transform
                crs = src.crs

                for geom, value in shapes(image, transform=transform):

                    if value > 0:

                        g = shape(geom)

                        all_geoms.append(g)
                        all_names.append(name)
                        all_areas.append(g.area)


    # 2 构建GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "filename": all_names,
            "pixel_area": all_areas
        },
        geometry=all_geoms,
        crs=crs
    )

    # 3 做并集
    union_geom = unary_union(gdf.geometry)

    # 4 拆分union结果
    if union_geom.geom_type == "Polygon":
        union_polys = [union_geom]
    else:
        union_polys = list(union_geom.geoms)

    final_geoms = []
    final_names = []
    final_area = []

    # 5 对每个union面找面积最大的原始面
    for poly in union_polys:

        intersect = gdf[gdf.geometry.intersects(poly)]

        if len(intersect) > 0:

            max_idx = intersect["pixel_area"].idxmax()

            name = intersect.loc[max_idx, "filename"]

        else:
            name = "unknown"

        final_geoms.append(poly)
        final_names.append(name)
        final_area.append(poly.area)

    # 6 输出
    final_gdf = gpd.GeoDataFrame(
        {
            "filename": final_names,
            "area": final_area
        },
        geometry=final_geoms,
        crs=crs
    )

    return final_gdf
