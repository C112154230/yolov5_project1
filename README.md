# YOLOv5 人與貓影像辨識專案

## 一、專案簡介
本專案使用 YOLOv5 進行影像辨識分析，目標為偵測 **人 (person)** 與 **貓 (cat)**。
專案包含資料處理、模型訓練、模型驗證與測試預測流程，並可完整重現實驗結果。

---
## 專案執行流程

1. 建立並啟動虛擬環境
2. 安裝必要套件
3. 準備資料集與 dataset.yaml
4. 執行模型訓練程式
5. 執行模型驗證
6. 使用訓練完成模型進行預測
---

## 二、環境安裝與配置（Environment Setup）

## 重要模組輸入 / 輸出說明

| 模組 | 輸入 | 輸出 |
|----|----|----|
| 資料處理 | 原始圖片、YOLO 標註 txt | 訓練/驗證資料集 |
| 模型訓練 | dataset.yaml、yolov5s.pt | best.pt、loss、mAP |
| 模型驗證 | best.pt、val dataset | Precision、Recall、mAP |
| 模型預測 | best.pt、測試圖片 | 標註後圖片 |
---

### 1. Python 與虛擬環境
建議使用 Python 3.10：

```bash
conda create -n yolov5_env python=3.10
conda activate yolov5_env

---
### 2.  啟動虛擬環境
----------------
# Windows
yolov5_env\Scripts\activate
# macOS/Linux
source yolov5_env/bin/activate
---
## 3.  安裝套件
---------------
pip install --upgrade pip
pip install ultralytics opencv-python matplotlib
---
## 4.  資料處理
---------------
YOLOv5_project/
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
├─ test_images/
---
## yolov5標註格式
----------------
<class> <x_center> <y_center> <width> <height>
---
## 資料集設定
-------------
train: "dataset/images/train"
val:   "dataset/images/val"
nc: 2
names: ['person', 'cat']
---
## 5.  模型訓練
----------------
from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    dataset_path = "dataset.yaml"
    model = YOLO("yolov5s.pt")

    model.train(
        data=dataset_path,
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        project="runs/train",
        name="person_cat"
    )

if __name__ == "__main__":
    freeze_support()
    main()
---
## 6.  模型驗證
-----------------
from ultralytics import YOLO

model = YOLO("runs/train/person_cat/weights/best.pt")
results = model.val()  # 驗證使用 train/val 資料集

# 印出 summary
for r in results.summary():
    print(r)
---
## 測試與預測
--------------
from ultralytics import YOLO

model = YOLO("runs/train/person_cat/weights/best.pt")
results = model.predict(source="test_images", save=True)
---
# 預測結果存放於 runs/detect/predict

Class      | Images | Instances | Precision | Recall | F1    | mAP50 | mAP50-95
-----------|--------|-----------|-----------|--------|-------|-------|---------
cat        | 125    | 611       | 0.5105    | 0.3961 | 0.4461| 0.3876| 0.2028
person     | 71     | 108       | 0.8890    | 0.8899 | 0.8894| 0.9160| 0.6831

mAP50 (整體) : 0.651830 / Rank X
