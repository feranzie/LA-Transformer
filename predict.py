import cv2
import torch
import timm
import numpy as np
from ultralytics import YOLO
from collections import deque
from torchvision import transforms
from typing import Dict, List, Tuple
from LATransformer.model import ClassBlock, LATransformer, LATransformerTest
from tqdm import tqdm

class PersonDetector:
    def __init__(self, model_size: str = "yolov8n.pt"):
        """Initialize YOLOv8 detector"""
        self.model = YOLO(model_size)
        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in frame
        Returns: List of [x1, y1, x2, y2] person detections
        """
        results = self.model(frame, classes=0)  # class 0 is person
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))
                    
        return detections

class ReIDModel:
    def __init__(self, model_path: str):
        """Initialize LA-Transformer ReID model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize LA-Transformer model
        vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
        self.model = LATransformerTest(vit_base, lmbd=8).to(self.device)
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract ReID features from a person crop"""
        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            # Normalize features
            fnorm = torch.norm(features, p=2, dim=1, keepdim=True) * np.sqrt(14)
            features_norm = features.div(fnorm.expand_as(features))
            return features_norm.view(-1).cpu()

class Track:
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], 
                 feature: torch.Tensor):
        self.id = track_id
        self.bbox = bbox
        self.features = deque([feature], maxlen=30)  # Keep last 30 features
        self.age = 0
        self.missed_frames = 0
    
    def update(self, bbox: Tuple[int, int, int, int], feature: torch.Tensor):
        self.bbox = bbox
        self.features.append(feature)
        self.age += 1
        self.missed_frames = 0
    
    def mark_missed(self):
        self.missed_frames += 1

class PersonTracker:
    def __init__(self, reid_model_path: str, 
                 yolo_model: str = "yolov8n.pt",
                 similarity_threshold: float = 0.6,
                 max_missed_frames: int = 3000):
        """
        Initialize tracker with both YOLOv8 and LA-Transformer
        """
        self.detector = PersonDetector(yolo_model)
        self.reid_model = ReIDModel(reid_model_path)
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_missed_frames = max_missed_frames
    
    def _compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> float:
        """Compute cosine similarity between two feature vectors"""
        return float(torch.nn.functional.cosine_similarity(feat1.unsqueeze(0), 
                                                         feat2.unsqueeze(0)))
    
    def _match_detections(self, detections: List[Tuple[int, int, int, int]], 
                         features: List[torch.Tensor]) -> Dict[int, Tuple[int, torch.Tensor]]:
        """Match detections with existing tracks"""
        matches = {}  # track_id -> (detection_idx, feature)
        unmatched_detections = set(range(len(detections)))
        
        # Match with existing tracks
        for track_id, track in self.tracks.items():
            if not unmatched_detections:
                break
                
            track_feat = track.features[-1]
            best_match = None
            best_similarity = 0
            
            for det_idx in unmatched_detections:
                similarity = self._compute_similarity(track_feat, features[det_idx])
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = det_idx
            
            if best_match is not None:
                matches[track_id] = (best_match, features[best_match])
                unmatched_detections.remove(best_match)
        
        return matches, unmatched_detections
    
    def update(self, frame: np.ndarray) -> Dict[int, Track]:
        """
        Update tracking for new frame
        Returns: Dictionary of active tracks
        """
        # Get detections
        detections = self.detector.detect(frame)
        
        # Extract features for all detections
        features = []
        valid_detections = []
        for bbox in detections:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:  # Skip invalid boxes
                continue
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            features.append(self.reid_model.extract_features(person_crop))
            valid_detections.append(bbox)
        
        # Match detections with existing tracks
        matches, unmatched_detections = self._match_detections(valid_detections, features)
        
        # Update matched tracks
        for track_id, (det_idx, feature) in matches.items():
            self.tracks[track_id].update(valid_detections[det_idx], feature)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self.tracks[self.next_id] = Track(
                self.next_id,
                valid_detections[det_idx],
                features[det_idx]
            )
            self.next_id += 1
        
        # Update unmatched tracks
        matched_track_ids = set(matches.keys())
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                self.tracks[track_id].mark_missed()
                if self.tracks[track_id].missed_frames > self.max_missed_frames:
                    del self.tracks[track_id]
        
        return self.tracks
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw tracking visualization on frame"""
        vis_frame = frame.copy()
        
        for track in self.tracks.values():
            x1, y1, x2, y2 = track.bbox
            
            # Color based on track age
            color = (0, 255 - min(track.age * 5, 255), min(track.age * 5, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and age
            text = f"ID: {track.id} Age: {track.age}"
            print(f'track id of {track.id} with {track.age} detected at x {(x1+x2)/2} and y {(y1+y2)/2}')
            
            cv2.putText(vis_frame, text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        return vis_frame

def main():
    
    # Initialize tracker
    tracker = PersonTracker(
        reid_model_path='model/net_best.pth',
        yolo_model='yolov8n.pt',
        similarity_threshold=0.6
    )
    
    # Initialize video capture
    cap = cv2.VideoCapture("/notebooks/20241003T122001.mkv")  # Use 0 for webcam or video file path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust the codec as needed (e.g., 'mp4v')
    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter('3idtesttrack.mp4', fourcc, 20.0, (output_width, output_height))  # Adjust filename and frame size
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update tracking
            tracks = tracker.update(frame)
            # Visualize
            vis_frame = tracker.draw_tracks(frame)

            # Display
            #cv2.imshow('Tracking', vis_frame)
            out.write(vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            #print(f'pbar is {pbar}')
            pbar.update(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()