import geopandas as gpd
import pandas as pd

# =========================

def join_table_xlsx(shp_path,xlsx_path):
    # =========================
    # 读取数据
    # =========================

    gdf = gpd.read_file(shp_path)

    df = pd.read_excel(xlsx_path)

    # =========================
    # 连接字段
    # =========================
    shp_field = "filename"
    xlsx_field = "image_name"

    # 检查字段是否存在
    if shp_field not in gdf.columns:
        raise ValueError(f"shp 中不存在字段: {shp_field}")
    if xlsx_field not in df.columns:
        raise ValueError(f"xlsx 中不存在字段: {xlsx_field}")

    # =========================
    # 类型统一（字符串+去空格）
    # =========================
    gdf[shp_field] = gdf[shp_field].astype(str).str.strip()
    df[xlsx_field] = df[xlsx_field].astype(str).str.strip()

    # =========================
    # 检查重复值
    # =========================
    dup_count = df.duplicated(subset=xlsx_field).sum()
    if dup_count > 0:
        print(f"警告：xlsx 中 {xlsx_field} 有 {dup_count} 个重复值，已保留第一条记录。")
        df = df.drop_duplicates(subset=xlsx_field, keep="first")

    # =========================
    # 执行连接
    # =========================
    gdf_joined = gdf.merge(df, left_on=shp_field, right_on=xlsx_field, how="left")

    # =========================
    # 保存结果
    # =========================
    return gdf_joined

