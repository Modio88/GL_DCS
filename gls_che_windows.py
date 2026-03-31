import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from gl import gl_cls, gl_detect

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class GlacialLakeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GL_detection/classification_system")
        self.root.geometry("1400x850")

        # =========================
        # 顶部控制区
        # =========================
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Input image folder").pack(side=tk.LEFT, padx=5)

        self.path_var = tk.StringVar(value="images")
        self.path_entry = ttk.Entry(control_frame, textvariable=self.path_var, width=70)
        self.path_entry.pack(side=tk.LEFT, padx=5)

        self.browse_btn = ttk.Button(control_frame, text="folder", command=self.select_folder)
        self.browse_btn.pack(side=tk.LEFT, padx=5)

        self.classify_btn = ttk.Button(control_frame, text="gl_classify", command=self.run_classify)
        self.classify_btn.pack(side=tk.LEFT, padx=15)

        self.detect_btn = ttk.Button(control_frame, text="gl_detect", command=self.run_detect)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar(value="Please select the folder containing the images, then click the button to proceed.")
        self.status_label = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # =========================
        # 图形显示区
        # =========================
        self.figure = plt.Figure(figsize=(14, 8), dpi=100)
        self.ax1 = self.figure.add_subplot(1, 2, 1)
        self.ax2 = self.figure.add_subplot(1, 2, 2)

        self.ax1.set_title("Image")
        self.ax1.axis("off")

        self.ax2.set_title("Result")
        self.ax2.axis("off")

        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.draw()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select an input image folder")
        if folder:
            self.path_var.set(folder)

    def run_classify(self):
        """点击 gl_classify 按钮时执行"""
        in_tif = self.path_var.get().strip()

        if not in_tif or not os.path.exists(in_tif):
            messagebox.showerror("Error", "Please select a valid folder for input images！")
            return

        self.status_var.set("running gl_cls，Please wait...")
        self.root.update_idletasks()

        try:
            rgb, shp = gl_cls(in_tif)
            cls_field = "top1_class"
            name_title = "Glacial lake type"
            self.plot_result(rgb, shp, cls_field, name_title, mode="class")
            self.status_var.set("gl_cls down")
        except Exception as e:
            messagebox.showerror("Error", f"gl_cls fail：\n{e}")
            self.status_var.set("gl_cls fail")

    def run_detect(self):
        """点击 gl_detect 按钮时执行"""
        in_tif = self.path_var.get().strip()

        if not in_tif or not os.path.exists(in_tif):
            messagebox.showerror("Error", "Please select a valid folder for input images！")
            return

        self.status_var.set("running gl_detect Please wait...")
        self.root.update_idletasks()

        try:
            rgb, shp = gl_detect(in_tif,band=8)
            cls_field = "Shape"  # 这里保留也可以，但 detect 模式下不会按它分类
            name_title = "Detected glacial lakes"
            self.plot_result(rgb, shp, cls_field, name_title, mode="detect")
            self.status_var.set("gl_detect down")
        except Exception as e:
            messagebox.showerror("Error", f"gl_detect fail：\n{e}")
            self.status_var.set("gl_detect fail")

    def plot_result(self, rgb, shp, cls_field, name_title, mode="class"):
        """在界面中绘制左图和右图
        mode="class"  -> 按字段分类着色
        mode="detect" -> 单一蓝色显示
        """
        import numpy as np

        # 清空旧图
        self.ax1.clear()
        self.ax2.clear()

        # =========================
        # 左图：原始影像
        # 自动兼容 CHW -> HWC
        # =========================
        if hasattr(rgb, "shape") and len(rgb.shape) == 3 and rgb.shape[0] in [3, 4]:
            rgb = np.transpose(rgb, (1, 2, 0))

        self.ax1.imshow(rgb)
        self.ax1.set_title("Image")
        self.ax1.axis("off")

        # =========================
        # 右图：结果图
        # =========================
        if shp is None or shp.empty:
            self.ax2.set_title(name_title)
            self.ax2.text(0.5, 0.5, "no output", ha='center', va='center', fontsize=16)
            self.ax2.axis("off")
            self.canvas.draw()
            return

        # ---------- detect 模式：单一蓝色 ----------
        if mode == "detect":
            shp.plot(
                ax=self.ax2,
                color="#4A90E2",  # 填充蓝色
                edgecolor="none",  # 边界可见（可改成 'none'）
                linewidth=0.5
            )

            # 可选：如果你想要一个简单图例，可以取消下面注释
            # patches = [mpatches.Patch(color="blue", label="Detected glacial lakes")]
            # self.ax2.legend(handles=patches, loc='best')

            self.ax2.set_title(name_title)
            self.ax2.axis("off")

            self.figure.tight_layout()
            self.canvas.draw()
            return

        # ---------- class 模式：按字段分类着色 ----------
        if cls_field not in shp.columns:
            self.ax2.set_title(name_title)
            self.ax2.text(0.5, 0.5, f"field_id {cls_field} not exist", ha='center', va='center', fontsize=14)
            self.ax2.axis("off")
            self.canvas.draw()
            return

        unique_classes = shp[cls_field].dropna().unique()
        cmap = plt.get_cmap("tab20")
        colors = {cls: cmap(i % 20) for i, cls in enumerate(unique_classes)}

        for cls, color in colors.items():
            subset = shp[shp[cls_field] == cls]
            subset.plot(ax=self.ax2, color=color, edgecolor='none')

        patches = [mpatches.Patch(color=color, label=str(cls)) for cls, color in colors.items()]
        if patches:
            self.ax2.legend(handles=patches, title=cls_field, loc='best')

        self.ax2.set_title(name_title)
        self.ax2.axis("off")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = GlacialLakeApp(root)
    root.mainloop()