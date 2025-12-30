
import cv2
import numpy as np
import os

def create_samples():
    base_dir = "sample_data/images"
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. Normal Traffic (Just cars)
    img_traffic = np.ones((640, 640, 3), dtype=np.uint8) * 200 # Light gray background
    # Draw road
    cv2.rectangle(img_traffic, (100, 0), (540, 640), (50, 50, 50), -1)
    # Draw cars (Rectangles)
    cars = [(200, 100, (0, 0, 255)), (350, 300, (255, 0, 0)), (220, 450, (0, 255, 0))]
    for x, y, col in cars:
        cv2.rectangle(img_traffic, (x, y), (x+80, y+140), col, -1)
    cv2.imwrite(f"{base_dir}/sample_traffic.jpg", img_traffic)
    
    # 2. Congestion (Many cars close together)
    img_cong = np.ones((640, 640, 3), dtype=np.uint8) * 200
    cv2.rectangle(img_cong, (100, 0), (540, 640), (50, 50, 50), -1)
    for i in range(10):
        y = 50 + i * 55
        x = 200 if i % 2 == 0 else 350
        cv2.rectangle(img_cong, (x, y), (x+80, y+50), (0, 0, 255), -1)
    cv2.imwrite(f"{base_dir}/sample_congestion.jpg", img_cong)

    # 3. Fire Incident (Orance/Red chaos)
    img_fire = np.ones((640, 640, 3), dtype=np.uint8) * 50 # Dark background
    # Draw road
    cv2.rectangle(img_fire, (100, 0), (540, 640), (30, 30, 30), -1)
    # Draw fire
    center = (320, 320)
    for i in range(50):
        radius = np.random.randint(10, 100)
        color = (np.random.randint(0, 50), np.random.randint(100, 255), np.random.randint(200, 255)) # BGR: Orange/Red-ish
        offset_x = np.random.randint(-50, 50)
        offset_y = np.random.randint(-50, 50)
        cv2.circle(img_fire, (center[0]+offset_x, center[1]+offset_y), radius, color, -1)
    cv2.imwrite(f"{base_dir}/sample_fire.jpg", img_fire)
    
    print("Sample images created in sample_data/images/")

if __name__ == "__main__":
    create_samples()
