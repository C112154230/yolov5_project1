from ultralytics import YOLO

def main():
    # 載入訓練好的模型
    model = YOLO("runs/train/person_cat4/weights/best.pt")

    # 驗證資料集
    results = model.val(data="C:/Users/ISLAB_309/Desktop/yl5/YOLOv5_project/dataset.yaml")

    # 取得每個類別的 summary
    summary_list = results.summary()  # list of dicts

    print("\n===== 每個類別的指標 =====")
    for cls_metrics in summary_list:
        print(f"Class: {cls_metrics['Class']}")
        print(f"  Images: {cls_metrics['Images']}")
        print(f"  Instances: {cls_metrics['Instances']}")
        print(f"  Precision (P): {cls_metrics['Box-P']:.4f}")
        print(f"  Recall (R): {cls_metrics['Box-R']:.4f}")
        print(f"  F1 score: {cls_metrics['Box-F1']:.4f}")
        print(f"  mAP50: {cls_metrics['mAP50']:.4f}")
        print(f"  mAP50-95: {cls_metrics['mAP50-95']:.4f}")
        print("----------------------------")

    # 計算整體 mAP50（平均所有類別）
    overall_map50 = sum(cls['mAP50'] for cls in summary_list) / len(summary_list)
    print(f"\n===== Private leaderboard (整體 mAP50) =====")
    print(f"mAP50: {overall_map50:.6f} / Rank X")  # Rank 可依照你比賽情況填

if __name__ == "__main__":
    main()
