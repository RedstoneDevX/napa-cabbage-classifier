# -*- coding: utf-8 -*-
"""蔬菜识别模型（白菜 / 花菜 / 青椒 / 其他）

    python3 predict.py 图片.jpg
    python3 predict.py 某个文件夹/

模型 = torchvision 的 MobileNetV3-Small(ImageNet 预训练，冻结) + 线性分类头。
分类头是 int8 量化后的 4×576 权重，拆成 weights/part0..3.txt（共约 3KB），不占仓库体积。
"""
import sys, os, glob, base64
import numpy as np, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["baicai", "huacai", "qingjiao", "other"]
LABELS = {"baicai": "白菜", "huacai": "花菜", "qingjiao": "青椒", "other": "其他"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
D, K = 576, 4


def _load_blob():
    parts = sorted(glob.glob(os.path.join(HERE, "weights", "part*.txt")))
    return "".join("".join(open(p).read().split()) for p in parts)


class VegetableNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(D, K)

    def forward(self, x):
        return self.head(self.backbone(x))


def load():
    raw = base64.b64decode(_load_blob())
    Wq = np.frombuffer(raw[:D * K], dtype=np.int8).reshape(K, D).astype(np.float32)
    scale = np.frombuffer(raw[D * K:D * K + K * 4], dtype=np.float32)
    b = np.frombuffer(raw[D * K + K * 4:], dtype=np.float32)
    net = VegetableNet()
    net.head.weight.data = torch.tensor(Wq * scale[:, None], dtype=torch.float32)
    net.head.bias.data = torch.tensor(b, dtype=torch.float32)
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
        print(f"{os.path.basename(f)}  ->  {LABELS[name]}  (置信度 {conf*100:.1f}%)")
