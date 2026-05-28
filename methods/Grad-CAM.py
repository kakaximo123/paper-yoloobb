import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO


class YOLOv11OBBGradCAM:
    def __init__(self, weight_path, target_layer_idx=-2, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(weight_path)
        self.model = self.yolo.model.to(self.device)
        self.model.eval()

        # 1. 开启所有参数的梯度
        for p in self.model.parameters():
            p.requires_grad = True

        # 2. 获取基本信息
        self.imgsz = getattr(self.yolo.overrides, "imgsz", 640) if hasattr(self.yolo, "overrides") else 640
        if isinstance(self.imgsz, (list, tuple)):
            self.imgsz = self.imgsz[0]
        self.names = self.model.names
        self.nc = len(self.names)

        # 3. 自动定位 Target Layer (特征层)
        self.target_layer = self.model.model[target_layer_idx]
        print(f"Target Layer: {self.target_layer._get_name()}")

        # 4. 自动定位 Detect Head 的分类分支
        self.detect_head = self.model.model[-1]
        self.cls_layers = self._find_cls_layers()

        # 5. 注册 Hooks
        self.activations = None
        self.gradients = None
        self.cls_outputs = []

        # Hook 特征层
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

        # Hook 分类头
        for layer in self.cls_layers:
            layer.register_forward_hook(self.cls_hook)

    def _get_out_channels(self, layer):
        """递归查找层的输出通道数，兼容 Conv2d, Sequential, Ultralytics Conv 等多种结构."""
        # 情况1: 是标准的 Conv2d
        if isinstance(layer, nn.Conv2d):
            return layer.out_channels

        # 情况2: 是 Ultralytics 的 Conv 模块 (有 .conv 属性)
        if hasattr(layer, "conv"):
            return self._get_out_channels(layer.conv)

        # 情况3: 是 Sequential 或 ModuleList (递归检查最后一个子模块)
        if isinstance(layer, (nn.Sequential, nn.ModuleList)):
            # 倒序遍历，找到第一个有输出通道的层
            for sub_layer in reversed(layer):
                ch = self._get_out_channels(sub_layer)
                if ch > 0:
                    return ch

        return -1

    def _find_cls_layers(self):
        """自动寻找 Detect 头中负责分类的卷积层。."""
        # 检查 cv2 分支
        if hasattr(self.detect_head, "cv2"):
            layer = self.detect_head.cv2[0]
            out_ch = self._get_out_channels(layer)
            if out_ch == self.nc:
                print(f"Found Class Branch: cv2 (Channels={out_ch})")
                return self.detect_head.cv2

        # 检查 cv3 分支
        if hasattr(self.detect_head, "cv3"):
            layer = self.detect_head.cv3[0]
            out_ch = self._get_out_channels(layer)
            if out_ch == self.nc:
                print(f"Found Class Branch: cv3 (Channels={out_ch})")
                return self.detect_head.cv3

        print("Warning: Auto-detection of CLS branch failed. Defaulting to cv3.")
        return getattr(self.detect_head, "cv3", [])

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def cls_hook(self, module, input, output):
        self.cls_outputs.append(output)

    def process_image(self, img_bgr):
        self.orig_h, self.orig_w = img_bgr.shape[:2]
        img_resized = cv2.resize(img_bgr, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).to(self.device).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor

    def generate(self, img_bgr, target_cls_idx=None):
        input_tensor = self.process_image(img_bgr)
        input_tensor.requires_grad = True

        self.model.zero_grad()
        self.cls_outputs = []

        # 1. 前向传播
        preds = self.model(input_tensor)

        if not self.cls_outputs:
            print("Error: 未捕获到分类头输出。")
            return img_bgr, np.zeros((self.orig_h, self.orig_w))

        # 2. 确定目标类别
        detect_output = preds[0] if isinstance(preds, (tuple, list)) else preds

        if detect_output.ndim == 3 and detect_output.shape[1] > 5:
            # OBB 输出通常是 [Batch, 4+1+nc, Anchors]
            cls_scores = detect_output[0, 5:, :]
            max_scores, max_idxs = torch.max(cls_scores, dim=0)
            best_idx = torch.argmax(max_scores)

            target_cls = max_idxs[best_idx].item()
            conf = max_scores[best_idx].item()
        else:
            target_cls = 0
            conf = 0.0

        if target_cls_idx is not None:
            target_cls = target_cls_idx

        print(f"Target Class: {self.names[target_cls]} | Conf: {conf:.4f}")

        # 3. 计算 Loss
        loss = 0
        for cls_out in self.cls_outputs:
            # cls_out: [1, nc, H, W]
            if cls_out.shape[1] != self.nc:
                continue  # 跳过不匹配的层

            target_map = cls_out[0][target_cls, :, :]
            loss += target_map.sum()

        loss.backward()

        # 4. 生成 CAM
        if self.gradients is None:
            print("Error: 梯度依然为 0。请尝试更换 target_layer_idx (如 -4)。")
            return img_bgr, np.zeros((self.orig_h, self.orig_w))

        gradients = self.gradients
        if isinstance(gradients, (list, tuple)):
            gradients = gradients[0]

        activations = self.activations
        if isinstance(activations, (list, tuple)):
            activations = activations[0]

        print(f"DEBUG >>> Gradients Max: {gradients.abs().max():.6f}")

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0, 0]

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam_resized = cv2.resize(cam, (self.orig_w, self.orig_h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        return overlay, cam_resized


if __name__ == "__main__":
    # 请替换为你的路径
    weight_path = r"D:\examples_project\python projects\self-enhance-all\ultralytics-main-3\ultralytics-main\runs\obb\train36\weights\best.pt"
    image_path = r"D:\研究生\会议论文\test\6.png"

    img = cv2.imread(image_path)
    if img is None:
        print("图片路径错误")
    else:
        grad_cam = YOLOv11OBBGradCAM(weight_path, target_layer_idx=-2)
        overlay, _ = grad_cam.generate(img)

        # cv2.imshow("Correct Grad-CAM", overlay)
        output_filename = os.path.join(r"D:\研究生\会议论文\test", "65.jpg")
        cv2.imwrite(output_filename, overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
