# -*- coding: utf-8 -*-
"""白菜识别模型（napa cabbage vs not）

    python3 predict.py 图片.jpg
    python3 predict.py 某个文件夹/

模型 = torchvision 的 MobileNetV3-Small(ImageNet 预训练，冻结) + 一个线性分类头。
分类头权重只有 577 个 fp16 数，存成 head_b64.txt（1.5KB base64），不占仓库体积。
"""
import sys, os, glob, base64
import numpy as np, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["baicai", "not_baicai"]          # 0=白菜, 1=不是白菜
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CabbageNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(576, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


def load(weights_path=os.path.join(HERE, "head_b64.txt")):
    blob = base64.b64decode(open(weights_path).read().strip())
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float32)   # 576 + 1
    w, b = arr[:576], arr[576]
    net = CabbageNet()
    net.head.weight.data = torch.tensor(np.vstack([np.zeros(576), w]), dtype=torch.float32)
    net.head.bias.data = torch.tensor([0.0, b], dtype=torch.float32)
    net.eval()
    prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return net, prep


@torch.no_grad()
def predict(path, net, prep):
    im = Image.open(path).convert("RGB")
    p = torch.softmax(net(prep(im).unsqueeze(0)), dim=1)[0]
    idx = int(p.argmax())
    return CLASSES[idx], float(p[idx])


if __name__ == "__main__":
    net, prep = load()
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(target, "*.*"))) if os.path.isdir(target) else [target]
    for f in files:
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")): continue
        name, conf = predict(f, net, prep)
        label = "白菜" if name == "baicai" else "不是白菜"
        print(f"{os.path.basename(f)}  ->  {label}  (置信度 {conf*100:.1f}%)")
