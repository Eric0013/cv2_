# =====================================================================
# 【地端專用】九宮格 AI 專屬數據集生成器 (make_grid_dataset.py)
# =====================================================================
import os
import cv2
import numpy as np

def create_fake_dataset():
    base_dir = "./local_grid_dataset"
    for s in ["train", "val"]:
        os.makedirs(f"{base_dir}/{s}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/{s}/labels", exist_ok=True)
    
    # 生成 15 張訓練圖，5 張驗證圖
    for mode in ["train", "val"]:
        count = 15 if mode == "train" else 5
        for img_idx in range(count):
            # 建立一個模擬比賽現場的黑灰底面板 (640x480)
            img = np.ones((480, 640, 3), dtype=np.uint8) * 40
            
            # 隨機微調亮度與反光，增加 AI 泛化能力
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=np.random.randint(-10, 15))
            
            label_lines = []
            # 畫出 3x3 圓形孔洞
            for row in range(3):
                for col in range(3):
                    # 圓心加上隨機抖動，模擬相機震動或斜拍
                    cx = int(160 + col * 160 + np.random.randint(-8, 8))
                    cy = int(120 + row * 120 + np.random.randint(-8, 8))
                    r = int(35 + np.random.randint(-3, 4))
                    
                    # 在圖上畫出白色的反光圓形孔洞
                    cv2.circle(img, (cx, cy), r, (240, 240, 240), -1)
                    
                    # 計算 YOLO 標準歸一化座標 (類別 0 代表 hole)
                    x_center = cx / 640.0
                    y_center = cy / 480.0
                    w_norm = (r * 2) / 640.0
                    h_norm = (r * 2) / 480.0
                    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            # 寫入影像與標記檔
            img_path = f"{base_dir}/{mode}/images/grid_{img_idx}.jpg"
            lbl_path = f"{base_dir}/{mode}/labels/grid_{img_idx}.txt"
            cv2.imwrite(img_path, img)
            with open(lbl_path, "w") as f:
                f.writelines(label_lines)
                
    # 建立 data.yaml 檔
    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: train/images
val: val/images

names:
  0: hole
"""
    with open(f"{base_dir}/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"🎉 專屬九宮格數據集創建成功！設定檔位於: {os.path.abspath(base_dir)}/data.yaml")

if __name__ == "__main__":
    create_fake_dataset()