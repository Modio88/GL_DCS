from ultralytics import YOLO
import os
import torch
import cv2
from pathlib import Path
import platform
from ultralytics.utils.plotting import Annotator  # 导入YOLO的标注器
def save_detection_results(results,  # YOLO模型的object检测结果
                           source_path,  # 源图像路径
                           save_dir='runs/detect/exp',  # 结果保存目录
                           save_txt=True,  # 是否保存TXT标签
                           save_img=True,  # 是否保存带标注的图像
                           view_img=False,  # 是否显示图像
                           hide_labels=False,  # 是否隐藏标签
                           hide_conf=False,  # 是否隐藏置信度
                           names=None,  # 类别名称字典
                           colors=None):  # 类别颜色字典
    """
    从results[0].boxes获取检测参数并保存结果
    """
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 获取单张图像的检测结果

    result = results[0]
    boxes1 = result.boxes
    im0 = cv2.imread(source_path)  # 原始图像
    img_path = Path(source_path)

    # 初始化标注器
    annotator = Annotator(im0.copy())

    # 获取图像归一化参数
    h, w = im0.shape[:2]
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化参数 (w, h, w, h)

    # 处理每个检测到的目标
    for j in range(len(boxes1)):
        box=boxes1[j]
        cls = int(box.cls)  # 类别ID
        conf = float(box.conf)  # 置信度
        xywh=box.xywhn[0].cpu().numpy().tolist()
        xyxy=box.xyxy[0].cpu().numpy().tolist()
        if save_txt:
            # 构建标签行
            line = (cls, *xywh, conf) if not hide_conf else (cls, *xywh)
            # 写入文件
            txt_path = save_dir / 'labels'
            txt_path.mkdir(parents=True, exist_ok=True)
            with open(f'{txt_path}/{img_path.stem}.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        # defect标注图像
        if save_img or view_img:
            label = None if hide_labels else (names[cls] if hide_conf else f'{names[cls]} {conf:.2f}')
            # 修改字体、线体
            annotator.sf = 1
            annotator.lw = 1
            annotator.tf = 1
            annotator.box_label(xyxy, label, color=colors(cls, True) if callable(colors) else colors[cls])
    # 处理标注后的图像
    im0 = annotator.result()

    # 显示图像
    if view_img:
        if platform.system() == 'Linux':
            cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(str(img_path), im0.shape[1], im0.shape[0])
        cv2.imshow(str(img_path), im0)
        cv2.waitKey(1)  # 1毫秒延迟

    # 保存带标注的图像
    if save_img:
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), im0)
        # print(f"保存结果到: {save_path}")

    return save_dir

def detect_gl(file_path,out_txt,band):
    if band==8:
        model = YOLO('detect_8.pt')
    else:
        model = YOLO('detect_3.pt')
    for filename in os.listdir(file_path):
        if filename.endswith('.tiff'):
            name = os.path.splitext(filename)[0]
            image_path = os.path.join(file_path, filename)
            lake = model.predict(image_path,
                                 imgsz=512,
                                 conf=0.412,  # 置信度阈值
                                 iou=0.7,  # IOU阈值
                                 device="0",  # 使用第0号GPU
                                 # save=True,  # 是否保存带标注的图像
                                 # save_txt=True,  # 是否保存TXT标签
                             )
            # img_dir = r"/mnt/e/study/glof/test/jpg/"  # 存放原始GeoTIFF的文件夹
            # img_path=os.path.join(img_dir,filename)
            save_detection_results(lake,  # YOLO模型的object检测结果
                                   image_path,  # 源图像路径
                                   save_dir=out_txt,  # 结果保存目录
                                   save_txt=True,  # 是否保存TXT标签
                                   save_img=False,  # 是否保存带标注的图像
                                   view_img=False,  # 是否显示图像
                                   hide_labels=False,  # 是否隐藏标签
                                   hide_conf=False,  # 是否隐藏置信度
                                   names={0: "lake"},  # 类别名称字典
                                   colors={0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)})  # 类别颜色字典
