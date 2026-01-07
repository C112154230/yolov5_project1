from ultralytics import YOLO
from multiprocessing import freeze_support
import os

def main():
    # 模型權重路徑
    weights_path = r"C:/Users/ISLAB_309/Desktop/yl5/runs/train/person_cat/weights/best.pt"
    
    # 測試圖片資料夾
    test_images_path = r"C:/Users/ISLAB_309/Desktop/yl5/test.images"
    
    # 輸出資料夾
    output_path = r"C:/Users/ISLAB_309/Desktop/yl5/runs/detect/test_results"

    # 確保輸出資料夾存在
    os.makedirs(output_path, exist_ok=True)

    # 載入模型
    model = YOLO(weights_path)

    # 執行檢測
    results = model.predict(
        source=test_images_path,
        imgsz=760,      # 輸入圖片尺寸
        conf=0.2,      # 信心閾值
        save=True,      # 是否保存圖片
        save_txt=True,  # 是否保存檢測框座標 txt
        project=output_path,  # 輸出路徑
        name="predict"        # 子資料夾名稱
    )

    print("測試完成！檢測結果已存於:", os.path.join(output_path, "predict"))

if __name__ == "__main__":
    freeze_support()  # Windows 必加
    main()
