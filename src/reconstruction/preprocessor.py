"""
Image preprocessing for face reconstruction.

Handles face detection, alignment, and normalization.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


class FacePreprocessor:
    """
    Preprocess images for 3D face reconstruction.
    
    Pipeline:
        1. Load image
        2. Detect face
        3. Align face
        4. Normalize lighting
        5. Resize to model input size
        6. Convert to tensor
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        device: Optional[str] = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height)
            device: Device for tensor operations
        """
        self.target_size = target_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Face detector (Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Eye detector for alignment
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
    
    def process(self, image_path: str) -> torch.Tensor:
        """
        Process image for model input.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Load image
        image = self._load_image(image_path)
        
        # Detect face
        face, bbox = self._detect_face(image)
        
        # Align face
        aligned = self._align_face(face, bbox)
        
        # Normalize
        normalized = self._normalize(aligned)
        
        # Resize
        resized = cv2.resize(normalized, self.target_size)
        
        # Convert to tensor
        tensor = self._to_tensor(resized)
        
        return tensor
    
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load image from path.
        
        Args:
            path: Image path
        
        Returns:
            Image as numpy array (BGR)
        """
        image = cv2.imread(path)
        
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        
        return image
    
    def _detect_face(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Detect face in image.
        
        Args:
            image: Input image (BGR)
        
        Returns:
            Tuple of (face ROI, bounding box)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        
        if len(faces) == 0:
            # Fallback: use center of image
            h, w = image.shape[:2]
            size = min(h, w) // 2
            x = (w - size) // 2
            y = (h - size) // 2
            bbox = (x, y, size, size)
        else:
            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            bbox = (x, y, w, h)
        
        # Extract face ROI
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        
        return face, bbox
    
    def _align_face(
        self,
        face: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Align face based on eye positions.
        
        Args:
            face: Face ROI
            bbox: Bounding box
        
        Returns:
            Aligned face
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray)
        
        if len(eyes) >= 2:
            # Sort eyes by x position
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Get eye centers
            left_eye = (eyes[0][0] + eyes[0][2] // 2, eyes[0][1] + eyes[0][3] // 2)
            right_eye = (eyes[1][0] + eyes[1][2] // 2, eyes[1][1] + eyes[1][3] // 2)
            
            # Calculate angle
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotate to align eyes horizontally
            h, w = face.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(face, M, (w, h))
            
            return aligned
        
        return face
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image lighting.
        
        Args:
            image: Input image
        
        Returns:
            Normalized image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Histogram equalization
        yuv = cv2.cvtColor(normalized, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        return normalized
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor.
        
        Args:
            image: Input image (BGR, uint8)
        
        Returns:
            Tensor [1, 3, H, W] normalized to [0, 1]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float() / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        tensor = (tensor - mean) / std
        
        # Transpose to [C, H, W]
        tensor = tensor.permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks.
        
        Args:
            image: Input image
        
        Returns:
            Landmark points (68, 2) or None
        """
        try:
            import dlib
            
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                points = np.array([[p.x, p.y] for p in landmarks.parts()])
                return points
        
        except ImportError:
            pass
        
        return None
