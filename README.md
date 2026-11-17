
## 仓库结构

```
.
├── predict.py          # 推理（读 weights/ 下的分片权重）
├── train.py            # 训练（特征提取 + 逻辑回归 + int8 量化）
├── crawl.py            # 抓白菜 / 非白菜图片
├── crawl_veg.py        # 抓花菜 / 青椒图片
├── clean_labels.py     # 用 CLIP 零样本自动查错标
├── requirements.txt
└── weights/
    ├── part0.txt       # 分类头 int8 权重分片（4 片拼起来才是完整 base64）
    ├── part1.txt
    ├── part2.txt
    └── part3.txt
```

分类头只有 3 KB，所以直接以 base64 文本存进仓库；主干 MobileNetV3-Small 在首次运行 import 时由 torchvision 自动下载。
