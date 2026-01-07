from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    dataset_path = r"C:/Users/ISLAB_309/Desktop/yl5/YOLOv5_project/dataset.yaml"

    model = YOLO("yolov5s.pt")  # 載入預訓練模型

    model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=8,
        workers=0,      # Windows 必加
        project="runs/train",
        name="person_cat"
    )

if __name__ == "__main__":
    freeze_support()
    main()
