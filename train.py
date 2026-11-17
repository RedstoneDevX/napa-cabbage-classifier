# -*- coding: utf-8 -*-
"""训练 4 分类：白菜 / 花菜 / 青椒 / 其他

MobileNetV3-Small 冻结特征 + 逻辑回归分类头

目录结构：
    data_clean/baicai/*.jpg        白菜
    veg_clean/huacai/*.jpg         花菜
    veg_clean/qingjiao/*.jpg       青椒
    data_clean/not_baicai/*.jpg    其他（含卷心菜、生菜、水果、汽车等，见 clean_list.json 白名单）

输出：head_b64.txt（int8 量化的分类头）、cabbage_model.pt、metrics.json
"""
import os, random, json, base64
import numpy as np, torch, torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image

random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
torch.set_num_threads(2)
CLASSES = ["baicai", "huacai", "qingjiao", "other"]
NAMES = {"baicai": "白菜", "huacai": "花菜", "qingjiao": "青椒", "other": "其他"}
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
D, K = 576, 4

keep = json.load(open("clean_list.json"))["keep"]
items = [(os.path.join("data_clean", "baicai", f), 0) for f in keep["baicai"]]
items += [(os.path.join("veg_clean", "huacai", f), 1) for f in sorted(os.listdir("veg_clean/huacai"))]
items += [(os.path.join("veg_clean", "qingjiao", f), 2) for f in sorted(os.listdir("veg_clean/qingjiao"))]
items += [(os.path.join("data_clean", "not_baicai", f), 3) for f in keep["not_baicai"]]

from collections import Counter
print("各类样本:", {NAMES[CLASSES[c]]: n for c, n in Counter([i[1] for i in items]).items()})

random.shuffle(items)
tr, va, cnt = [], [], {i: 0 for i in range(4)}
for p, c in items:
    (va if cnt[c] % 5 == 0 else tr).append((p, c)); cnt[c] += 1
print("train", len(tr), "val", len(va))

m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()
backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
for p in backbone.parameters(): p.requires_grad_(False)
prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                           transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

@torch.no_grad()
def feats(paths):
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        out.append(backbone(prep(im).unsqueeze(0)).numpy()[0])
    return np.array(out, dtype=np.float32)

Xtr = feats([p for p, _ in tr]); ytr = np.array([c for _, c in tr])
Xva = feats([p for p, _ in va]); yva = np.array([c for _, c in va])
clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
pred = clf.predict(Xva)
print("验证集准确率: %.4f" % accuracy_score(yva, pred))
print(classification_report(yva, pred, target_names=[NAMES[c] for c in CLASSES], digits=4))
print("混淆矩阵(行=真实,列=预测):"); print(confusion_matrix(yva, pred))

W = clf.coef_.astype(np.float32); b = clf.intercept_.astype(np.float32)
# int8 量化打包：权重 + 每行 scale + 偏置
scale = np.abs(W).max(axis=1, keepdims=True) / 127.0
Wq = np.round(W / scale).astype(np.int8)
blob = Wq.tobytes() + scale.ravel().astype(np.float32).tobytes() + b.tobytes()
open("head_b64.txt", "w").write(base64.b64encode(blob).decode())

class Net(nn.Module):
    def __init__(self, backbone, W, b):
        super().__init__(); self.backbone = backbone
        self.head = nn.Linear(W.shape[1], W.shape[0])
        self.head.weight.data = torch.from_numpy(W).float()
        self.head.bias.data = torch.from_numpy(b).float()
    def forward(self, x): return self.head(self.backbone(x))

torch.save({"state_dict": Net(backbone, W, b).state_dict(), "classes": CLASSES,
            "img_size": 224, "mean": MEAN, "std": STD}, "cabbage_model.pt")
json.dump({"val_acc": float(accuracy_score(yva, pred)), "n_train": len(tr), "n_val": len(va)},
          open("metrics.json", "w"), ensure_ascii=False, indent=2)
print("已保存 head_b64.txt 与 cabbage_model.pt")
