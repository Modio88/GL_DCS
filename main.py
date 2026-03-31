from gl import gl_cls, gl_detect
in_tif = "images"  # 输入影像
task = 'class'
band=8
if task == "class":
    rgb, shp = gl_cls(in_tif)
    cls_field = "top1_class"
    name_title = "Glacial lake type"
else:
    rgb, shp = gl_detect(in_tif,band)
    cls_field = "Shape"
    name_title = "Detected glacial lakes"

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
from rasterio.plot import show

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 绘图
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# 左图：原始影像
axs[0].imshow(rgb)
axs[0].set_title("Image")
axs[0].axis('off')

# 右图：根据任务类型绘制 shp
if task == "class":
    # 分类情景：按字段着色（保持不变）
    unique_classes = shp[cls_field].unique()
    cmap = plt.get_cmap("tab20")
    colors = {cls: cmap(i % 20) for i, cls in enumerate(unique_classes)}

    for cls, color in colors.items():
        subset = shp[shp[cls_field] == cls]
        subset.plot(ax=axs[1], color=color, edgecolor='none')

    # 自定义图例
    patches = [mpatches.Patch(color=color, label=str(cls)) for cls, color in colors.items()]
    axs[1].legend(handles=patches, title=cls_field)

else:
    # 检测情景：统一蓝色显示，不按字段分类
    shp.plot(ax=axs[1], color='#4A90E2', edgecolor='none')

# 右图标题
axs[1].set_title(name_title)
axs[1].axis('off')

plt.tight_layout()
plt.show()

# cd /mnt/e/study/glof/glacial_lake_system
