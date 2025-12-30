
import cv2
import numpy as np

def prepare_image(image_path, size=(640, 640)):
    """
    Reads and resizes image for model consistency.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, size)
    return img_resized

def normalize_frame(frame):
    """
    Normalize pixel values if needed for custom processing.
    """
    return frame.astype(np.float32) / 255.0
