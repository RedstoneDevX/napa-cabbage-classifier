# -*- coding: utf-8 -*-
"""训练：MobileNetV3-Small 冻结特征 + 逻辑回归分类头

数据目录结构：
    data_clean/baicai/*.jpg
    data_clean/not_baicai/*.jpg
    clean_list.json   ->  {"keep": {"baicai": [...], "not_baicai": [...]}}

输出：
    白菜识别模型.pt      （完整 PyTorch 权重）
    head_b64.txt         （仅分类头的 fp16 base64，1.5KB）
    metrics.json         （指标）
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
CLASSES = ["baicai", "not_baicai"]; NAMES = {"baicai": "白菜", "not_baicai": "不是白菜"}
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]

keep = json.load(open("clean_list.json"))["keep"]
items = [(os.path.join("data_clean", c, f), ci) for ci, c in enumerate(CLASSES) for f in keep[c]]
random.shuffle(items)
tr, va, cnt = [], [], {0: 0, 1: 0}
for p, c in items:
    (va if cnt[c] % 5 == 0 else tr).append((p, c)); cnt[c] += 1
print("train", len(tr), "val", len(va))

w = MobileNet_V3_Small_Weights.IMAGENET1K_V1
m = mobilenet_v3_small(weights=w).eval()
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
acc = accuracy_score(yva, pred)
print("验证集准确率: %.4f" % acc)
print(classification_report(yva, pred, target_names=[NAMES[c] for c in CLASSES], digits=4))
print(confusion_matrix(yva, pred))

W = clf.coef_; b = clf.intercept_
if W.shape[0] == 1:
    W = np.vstack([np.zeros_like(W), W]); b = np.array([0.0, b[0]], dtype=np.float32)

# 保存分类头（fp16 base64，predict.py 直接读这个）
blob = np.concatenate([W[1].astype(np.float16), np.array([b[1]], dtype=np.float16)])
open("head_b64.txt", "w").write(base64.b64encode(blob.tobytes()).decode())

class CabbageNet(nn.Module):
    def __init__(self, backbone, W, b):
        super().__init__(); self.backbone = backbone
        self.head = nn.Linear(W.shape[1], W.shape[0])
        self.head.weight.data = torch.from_numpy(W).float()
        self.head.bias.data = torch.from_numpy(b).float()
    def forward(self, x): return self.head(self.backbone(x))

net = CabbageNet(backbone, W, b).eval()
torch.save({"state_dict": net.state_dict(), "classes": CLASSES, "img_size": 224,
            "mean": MEAN, "std": STD, "val_acc": float(acc)}, "cabbage_model.pt")
json.dump({"val_acc": float(acc), "n_train": len(tr), "n_val": len(va),
           "cabbage_recall": float(((yva == 0) & (pred == 0)).sum() / max(1, (yva == 0).sum())),
           "cabbage_precision": float(((pred == 0) & (yva == 0)).sum() / max(1, (pred == 0).sum()))},
          open("metrics.json", "w"), ensure_ascii=False, indent=2)
print("已保存 cabbage_model.pt 与 head_b64.txt")
