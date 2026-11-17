# -*- coding: utf-8 -*-
"""用 CLIP 零样本判断每张图"到底是什么"，清洗抓图时混进来的错标样本

输入：data_clean/baicai/*.jpg 与 data_clean/not_baicai/*.jpg
输出：clean_list.json   {"keep": {...}, "report": {...}}
"""
import os, json, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

torch.set_num_threads(2)

GROUPS = {
 "白菜": ["一颗完整的大白菜", "大白菜特写照片", "大白菜的叶子", "napa cabbage", "chinese cabbage"],
 "其他蔬菜": ["一盘生菜", "一颗卷心菜", "西兰花", "菠菜", "青菜叶子", "番茄", "土豆"],
 "水果": ["水果"], "肉蛋奶": ["鸡蛋和牛奶"], "菜肴": ["做好的菜"],
 "人物": ["一个人"], "动物": ["一只猫"], "汽车": ["一辆汽车"], "家具": ["沙发和椅子"],
 "电器": ["笔记本电脑"], "文字图表": ["带文字的表格或海报"], "电路图": ["电路原理图"],
 "风景": ["城市风景"], "建筑": ["一栋建筑"], "天空": ["天空"],
}
prompts, owner = [], []
for g, ps in GROUPS.items():
    for p in ps: prompts.append(f"一张{p}的照片"); owner.append(g)

m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
T = proc(text=prompts, return_tensors="pt", padding=True)


@torch.no_grad()
def group_probs(paths, bs=8):
    res = []
    for i in range(0, len(paths), bs):
        ims = []
        for pth in paths[i:i + bs]:
            try: ims.append(Image.open(pth).convert("RGB"))
            except Exception: ims.append(Image.new("RGB", (224, 224)))
        x = proc(images=ims, return_tensors="pt")
        o = m(input_ids=T["input_ids"], attention_mask=T["attention_mask"], pixel_values=x["pixel_values"])
        res += o.logits_per_image.softmax(dim=-1).tolist()
    return res


report, keep = {}, {}
for cls in ["baicai", "not_baicai"]:
    d = os.path.join("data_clean", cls)
    files = sorted(os.listdir(d)); paths = [os.path.join(d, f) for f in files]
    probs = group_probs(paths)
    groups = [owner[max(range(len(p)), key=lambda j: p[j])] for p in probs]
    kept, dropped = [], []
    for f, g in zip(files, groups):
        if cls == "baicai":
            (kept if g == "白菜" else dropped).append((f, g))
        else:
            (kept if g != "白菜" else dropped).append((f, g))
    keep[cls] = kept
    report[cls] = {"总": len(files), "保留": len(kept), "剔除": len(dropped),
                   "剔除原因分布": {g: sum(1 for _, x in dropped if x == g) for g in set(x for _, x in dropped)}}
    print(cls, json.dumps(report[cls], ensure_ascii=False))

json.dump({"keep": {k: [f for f, _ in v] for k, v in keep.items()}, "report": report},
          open("clean_list.json", "w"), ensure_ascii=False, indent=1)
