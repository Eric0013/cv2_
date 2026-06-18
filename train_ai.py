import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def generate_advanced_calibration_dataset():
    base_dir = "./auto_calib_dataset"
    print("🔄 [1/3] 正在原地生成 350 張高動態環境、多雜訊幾何校準演練圖...")
    
    for s in ["train", "val"]:
        os.makedirs(f"{base_dir}/{s}/images", exist_ok=True)
        os.makedirs(f"{base_dir}/{s}/labels", exist_ok=True)

    for mode in ["train", "val"]:
        count = 300 if mode == "train" else 50
        for img_idx in range(count):
            
            # 1. 模擬比賽面板底色與光影漸變
            bg_base = np.random.randint(40, 75)
            img = np.ones((480, 640, 3), dtype=np.uint8) * bg_base
            noise = np.random.normal(0, 3, img.shape).astype(np.int16)
            img = cv2.add(img.astype(np.int16), noise)
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            label_lines = []
            
            # 2. 隨機產生相機視角斜切、縮放與平移
            scale_x = np.random.uniform(0.85, 1.12)
            scale_y = np.random.uniform(0.85, 1.12)
            shift_x = np.random.randint(-25, 25)
            shift_y = np.random.randint(-15, 25)

            # 3. 4 排、每排 3 孔的 3x4 空間結構解算
            for row in range(4):
                for col in range(3):
                    if row == 0:
                        base_cx = 283 + (col - 1) * 54
                        base_cy = 77
                        r_min, r_max = 7, 11
                    elif row == 1:
                        base_cx = 148 + col * 175
                        base_cy = 110
                        r_min, r_max = 20, 26
                    elif row == 2:
                        base_cx = 143 + col * 176
                        base_cy = 222
                        r_min, r_max = 25, 33
                    else:
                        base_cx = 140 + col * 181
                        base_cy = 378
                        r_min, r_max = 32, 45

                    cx = int(base_cx * scale_x + shift_x)
                    cy = int(base_cy * scale_y + shift_y)
                    r = int(np.random.randint(r_min, r_max + 1) * ((scale_x + scale_y) / 2))
                    
                    cx = max(r + 3, min(640 - r - 3, cx))
                    cy = max(r + 3, min(480 - r - 3, cy))

                    # 孔洞內部為深色黑洞
                    hole_color = np.random.randint(5, 30)
                    cv2.circle(img, (cx, cy), r, (hole_color, hole_color, hole_color), -1)
                    
                    # 模擬圓孔邊緣切面反光
                    if np.random.rand() > 0.4:
                        arc_brightness = np.random.randint(160, 245)
                        cv2.circle(img, (cx, cy), r, (arc_brightness, arc_brightness, arc_brightness), 1, lineType=cv2.LINE_AA)

                    # 換算標準化 YOLO 座標
                    x_center = cx / 640.0
                    y_center = cy / 480.0
                    w_norm = (r * 2) / 640.0
                    h_norm = (r * 2) / 480.0
                    label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            # 4. 背景高亮白色文字雜訊惡意干擾
            if np.random.rand() > 0.3:
                txt_color = np.random.randint(210, 255)
                random_txt = np.random.choice(["10", "TEL", "AI", "ROBO", "88", "1"])
                tx = np.random.randint(50, 580)
                ty = np.random.randint(30, 450)
                cv2.putText(img, random_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                            np.random.uniform(0.4, 0.8), (txt_color, txt_color, txt_color), 2, cv2.LINE_AA)

            cv2.imwrite(f"{base_dir}/{mode}/images/calib_{img_idx}.jpg", img)
            with open(f"{base_dir}/{mode}/labels/calib_{img_idx}.txt", "w") as f:
                f.writelines(label_lines)

    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: train/images
val: val/images
names:
  0: hole
"""
    with open(f"{base_dir}/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"✅ 增強型幾何校準數據集部署完畢！")

def main():
    generate_advanced_calibration_dataset()
    chosen_device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"\n🔄 [2/3] 硬體加速確認: 使用 {str(chosen_device).upper()} (RTX 3060)")

    print("\n🔄 [3/3] 載入 YOLOv8 核心，開啟 350 張高仿真幾何圖集深度演練...")
    model = YOLO('yolov8n.pt')
    model.train(
        data="./auto_calib_dataset/data.yaml",
        epochs=80,               
        imgsz=640,
        device=chosen_device,
        degrees=12.0,            
        scale=0.15,              
        translate=0.1,           
        shear=4.0,               
        perspective=0.0005,      
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.6,               
        mosaic=0.0,              
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        project='Taiko_Shohei',
        name='Grid_Hole_Pro'
    )
    print("\n=====================================================================")
    print(" 🎉 【重訓完畢】全新抗雜訊、自校準的 12 孔大腦權重已出爐！")
    print("=====================================================================")

if __name__ == "__main__":
    main()