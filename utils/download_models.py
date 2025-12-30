
import requests
import os

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded to {save_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Models to download
models = {
    "https://github.com/godrock44/YOLO-V8-FIRE-DETECTION/raw/main/best.pt": "models/yolo/fire_smoke.pt",
    "https://github.com/shubhankar-shandilya-india/Accident-Detection-Model/raw/main/yolov8s.pt": "models/yolo/accident.pt"
}

for url, path in models.items():
    download_file(url, path)
