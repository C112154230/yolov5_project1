from ultralytics import YOLO

model = YOLO(r"C:/Users/ISLAB_309/Desktop/yl5/runs/train/person_cat/weights/best.pt")

model.predict(
    source=r"C:/Users/ISLAB_309/Desktop/yl5/test.images",
    save=True,
    conf=0.1,
    iou=0.3,
    project="runs/detect",
    name="person_cat_test"
)
