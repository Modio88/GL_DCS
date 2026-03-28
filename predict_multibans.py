import os
import torch
import pandas as pd
import tifffile
import numpy as np

from ultralytics26.engine.predictor import BasePredictor
from ultralytics26.engine.results import Results
from ultralytics26.utils import DEFAULT_CFG, ops

# ===== 导入你训练时的 transforms =====
from ultralytics26.data.dataset import (
    create_multiband_val_transform
)


class MultiBandClassificationPredictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "classify"
        self.all_rows = []

        # self.hook_registered = False
        # self.gradcam = None
        # register_feature_hooks(self.model)

    def setup_source(self, source):
        """只设置 source，不用官方 classify_transforms"""
        super().setup_source(source)

        mean = [0.2498, 0.2353, 0.2213, 0.2098, 0.4473, 0.5433, 0.5275, 0.5570]
        std = [0.1885, 0.1868, 0.1856, 0.1420, 0.1499, 0.1271, 0.2060, 0.2426]

        self.transforms = create_multiband_val_transform(
            self.args, mean, std
        )

    def preprocess(self, img):
        """
        img: list[np.ndarray]
        """
        # if not self.hook_registered:
        #     register_feature_hooks(self.model)
        #
        #     # target_layer = self.model.model.model[9]
        #     #
        #     # self.gradcam = GradCAM(self.model, target_layer)
        # #
        #     self.hook_registered = True
        #
        #     print("Feature hooks + GradCAM registered")
        imgs = []

        for im in img:
            # 读取 8 通道 tiff
            if isinstance(im, str):
                im = tifffile.imread(im)

            im = im.astype(np.float32) / 255.0
            # print(im.shape)
            # (H,W,C) → (C,H,W)
            if im.ndim == 3:
                im = np.transpose(im, (2, 0, 1))

            im = self.transforms(im)
            imgs.append(im)

        img = torch.stack(imgs, dim=0).to(self.model.device)
        # print(self.model.model)

        return img.half() if self.model.fp16 else img.float()

    def postprocess(self, preds, img, orig_imgs):

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        preds = preds[0] if isinstance(preds, (list, tuple)) else preds

        results = []
        rows = []  # 用来存Excel数据

        for i, (pred, orig_img, img_path) in enumerate(zip(preds, orig_imgs, self.batch[0])):

            probs = pred.softmax(0)

            # 取 top5
            top5_prob, top5_idx = torch.topk(probs, 5)

            # 类别名称
            top5_names = [self.model.names[i] for i in top5_idx.tolist()]

            img_name = os.path.splitext(os.path.basename(img_path))[0]

            # 构建一行Excel数据
            row = {"image_name": img_name}

            for k, (name, prob) in enumerate(zip(top5_names, top5_prob.tolist()), start=1):
                row[f"top{k}_class"] = name
                row[f"top{k}_prob"] = round(prob, 4)

            self.all_rows.append(row)


            # 保留原有返回
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, probs=pred)
            )


        return results

    def save_excel(self):
        if not self.all_rows:
            print("没有分类结果可保存。")
            return

        save_dir = self.args.project
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, "cls_gl.xlsx")
        df = pd.DataFrame(self.all_rows)
        df.to_excel(save_path, index=False)