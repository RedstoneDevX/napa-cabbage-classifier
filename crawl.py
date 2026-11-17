# -*- coding: utf-8 -*-
"""从 Bing 图片搜索批量抓图，作为数据集候选池（后续还要清洗）"""
import re, os, time
import requests
from concurrent.futures import ThreadPoolExecutor

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
H = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}
OUT = "data_raw"

POS = ["大白菜", "大白菜 整颗", "大白菜 特写", "白菜 蔬菜", "黄心大白菜", "大白菜 菜地",
       "napa cabbage", "chinese cabbage", "大白菜 白底"]
NEG = ["卷心菜", "生菜", "西兰花", "菠菜", "油麦菜", "番茄", "土豆", "胡萝卜", "黄瓜", "青椒", "洋葱", "茄子",
       "南瓜", "芹菜", "豆角", "蘑菇", "苹果", "香蕉", "篮球", "笔记本电脑", "汽车", "猫", "狗", "手机", "椅子",
       "矿泉水瓶", "书包", "运动鞋", "面包", "蛋糕", "西瓜", "草莓", "葡萄", "橘子", "梨", "玉米", "大蒜", "生姜",
       "鸡蛋", "牛奶盒", "台灯", "雨伞", "键盘", "显示器", "自行车", "飞机", "花盆", "沙发", "枕头", "茶杯"]


def bing_urls(q, pages=2):
    urls = []
    for p in range(pages):
        try:
            r = requests.get("https://cn.bing.com/images/search",
                             params={"q": q, "first": 1 + p * 35, "count": 35},
                             headers=H, timeout=15)
            txt = r.text
        except Exception as e:
            print("search fail", q, e); continue
        for m in re.findall(r'murl(?:&quot;|"):(?:&quot;|")([^&"]+)', txt):
            u = m.replace("\\u0026", "&")
            if u.startswith("http"): urls.append(u)
        time.sleep(0.4)
    return urls


def dl(args):
    url, path = args
    if os.path.exists(path): return True
    try:
        r = requests.get(url, headers=H, timeout=12, stream=True)
        if r.status_code != 200: return False
        data = r.raw.read(4 * 1024 * 1024)
        if len(data) < 6000: return False
        with open(path, "wb") as f: f.write(data)
        return True
    except Exception:
        return False


def collect(label, queries, pages=2):
    d = os.path.join(OUT, label)
    os.makedirs(d, exist_ok=True)
    all_urls = []
    for q in queries:
        us = bing_urls(q, pages)
        print(f"[{label}] {q}: {len(us)}")
        all_urls += us
    all_urls = list(dict.fromkeys(all_urls))
    tasks = [(u, os.path.join(d, f"{label}_{i:04d}.jpg")) for i, u in enumerate(all_urls)]
    with ThreadPoolExecutor(32) as ex:
        res = list(ex.map(dl, tasks))
    print(f"[{label}] urls={len(all_urls)} downloaded={sum(res)}")


if __name__ == "__main__":
    collect("baicai", POS, pages=3)
    collect("not_baicai", NEG, pages=2)
