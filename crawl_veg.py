# -*- coding: utf-8 -*-
"""批次抓花菜 / 青椒候选图（用于扩充分类）"""
import re, os, time, hashlib
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
H = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}

Q = {
 "huacai": ["花菜", "白花菜", "菜花 蔬菜", "花椰菜", "花菜 整颗", "cauliflower", "花菜 白底"],
 "qingjiao": ["青椒", "青椒 蔬菜", "甜椒 绿色", "青椒 一堆", "green bell pepper", "青椒 白底"],
}


def bing(q, pages=3):
    urls = []
    for p in range(pages):
        try:
            r = requests.get("https://cn.bing.com/images/search",
                             params={"q": q, "first": 1 + p * 35, "count": 35}, headers=H, timeout=15)
            for m in re.findall(r'murl(?:&quot;|\"):(?:&quot;|\")([^&\"]+)', r.text):
                u = m.replace("\\u0026", "&")
                if u.startswith("http"): urls.append(u)
        except Exception as e:
            print("fail", q, e)
        time.sleep(0.3)
    return urls


def dl(args):
    url, path = args
    try:
        r = requests.get(url, headers=H, timeout=12, stream=True)
        if r.status_code != 200: return None
        data = r.raw.read(3 * 1024 * 1024)
        if len(data) < 6000: return None
        with open(path, "wb") as f: f.write(data)
        return path
    except Exception:
        return None


def build(label, queries):
    d = f"veg_raw/{label}"
    os.makedirs(d, exist_ok=True)
    urls = []
    for q in queries:
        u = bing(q); print(f"[{label}] {q}: {len(u)}"); urls += u
    urls = list(dict.fromkeys(urls))
    tasks = [(u, os.path.join(d, f"{label}_{i:04d}.jpg")) for i, u in enumerate(urls)]
    with ThreadPoolExecutor(32) as ex:
        res = [r for r in ex.map(dl, tasks) if r]
    print(f"[{label}] 下载 {len(res)} / {len(urls)}")


if __name__ == "__main__":
    for k, v in Q.items():
        build(k, v)
