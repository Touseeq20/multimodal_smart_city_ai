import cv2
import os
from ultralytics import YOLO
import numpy as np

class IncidentDetector:
    def __init__(self, model_path='yolov8x.pt', fire_model_path='models/yolo/fire_smoke.pt'):
        """
        Initialize Multi-Model AI Engine.
        1. YOLOv8x: SOTA General Detection & Tracking.
        2. Specialized Fire/Smoke Model: Dedicated neural engine for fire safety.
        """
        print(f"Loading SOTA Hybrid Intelligence Engine...")
        self.main_model = YOLO(model_path)
        
        # Load specialized fire model if exists, fallback to main
        if os.path.exists(fire_model_path):
            print(f"Loading Specialized Fire/Smoke Neural Engine...")
            self.fire_model = YOLO(fire_model_path)
        else:
            print(f"Warning: Specialized Fire model not found. Falling back to heuristics.")
            self.fire_model = None

        self.vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        self.person_class = 0
        self.history = {} # Stores positions for speed/stagnancy analysis

    def detect(self, image, track=False):
        """
        Dual-Inference Pipeline.
        """
        # 1. Main Detection (Vehicles, People, Tracking)
        if track:
            main_results = self.main_model.track(image, conf=0.15, persist=True, tracker="bytetrack.yaml")
        else:
            main_results = self.main_model(image, conf=0.15)
        
        # 2. Specialized Fire/Smoke Inference
        fire_results = None
        if self.fire_model:
            fire_results = self.fire_model(image, conf=0.3) # More confident for specialized task
        
        return main_results[0], (fire_results[0] if fire_results else None)

    def analyze_incident(self, main_result, fire_result=None):
        """
        Hyper-sensitive Hybrid Intelligence Engine.
        Combined Neural Detection + Geometric Reasoning.
        """
        boxes = main_result.boxes
        vehicles = []
        persons = []
        
        # Track IDs
        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            tid = track_ids[i]
            
            # Center calculation
            cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
            
            if tid is not None:
                if tid not in self.history:
                    self.history[tid] = []
                self.history[tid].append((cx, cy))
                if len(self.history[tid]) > 15: self.history[tid].pop(0)

            if cls_id in self.vehicle_classes:
                vehicles.append({'box': xyxy, 'conf': conf, 'id': tid, 'cls': cls_id, 'center': (cx, cy)})
            elif cls_id == self.person_class:
                persons.append(xyxy)
        
        incident_type = "Normal Traffic"
        details = {
            "vehicle_count": len(vehicles), 
            "person_count": len(persons),
            "severity": "LOW",
            "flags": []
        }
        
        # 1. SPECIALIZED FIRE/SMOKE DETECTION (Neural)
        if fire_result and len(fire_result.boxes) > 0:
            # Check fire model classes (typically 0: fire, 1: smoke)
            fire_conf = float(fire_result.boxes.conf[0])
            if fire_conf > 0.35:
                incident_type = "Critical: Active Fire/Smoke Detected"
                details["severity"] = "HIGH"
                details["flags"].append(f"Specialized Model: High Confidence Fire ({fire_conf:.2f})")

        # 2. CONGESTION & MOVEMENT
        stagnant_count = 0
        for v in vehicles:
            tid = v['id']
            if tid and len(self.history.get(tid, [])) >= 10:
                history = self.history[tid]
                dist = np.sqrt((history[-1][0]-history[0][0])**2 + (history[-1][1]-history[0][1])**2)
                if dist < 10: stagnant_count += 1
        
        if stagnant_count >= 8:
            incident_type = "Severe Traffic Gridlock"
            details["severity"] = "HIGH"
            details["flags"].append(f"Gridlock Alert: {stagnant_count} vehicles immobilized")
        elif stagnant_count >= 4:
            incident_type = "Traffic Congestion"
            details["severity"] = "MEDIUM"
            details["flags"].append(f"Movement Analysis: {stagnant_count} vehicles blocked")

        # 3. ADVANCED CRASH & OVERTURNED DETECTION (Geometric Intelligence)
        crash_indicators = 0
        overturned_detected = False
        
        for i, v1 in enumerate(vehicles):
            b1 = v1['box']
            w, h = b1[2]-b1[0], b1[3]-b1[1]
            aspect_ratio = w / h if h > 0 else 0
            
            # Geometric Alert: Overturned Vehicles
            # Normally oriented vehicles have a predictable AR from CCTV angles.
            # Cars (2), Trucks (7), Buses (5)
            if v1['cls'] in [2, 5, 7]:
                # If a vehicle is flipped, its bounding box ratio usually becomes extreme:
                # Vertical flipped: AR < 0.65
                # Sideways flipped (squashed profile): AR > 3.5
                if aspect_ratio > 3.5 or aspect_ratio < 0.65:
                    overturned_detected = True
                    details["flags"].append(f"Geometric Alert: Overturned {self.main_model.names[v1['cls']]} detected!")

            for j in range(i + 1, len(vehicles)):
                iou = self.compute_iou(b1, vehicles[j]['box'])
                if iou > 0.12: crash_indicators += 1

        if overturned_detected or crash_indicators > 0 or (len(vehicles) > 0 and len(persons) >= 2):
            if incident_type == "Normal Traffic" or "Congestion" in incident_type:
                incident_type = "Critical Traffic Accident"
                if overturned_detected:
                    incident_type = "Severe Emergency: Overturned Vehicle"
                details["severity"] = "HIGH"
            
        return incident_type, details

    @staticmethod
    def compute_iou(box1, box2):
        """
        Compute Intersection over Union (IoU) of two bounding boxes.
        box: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
