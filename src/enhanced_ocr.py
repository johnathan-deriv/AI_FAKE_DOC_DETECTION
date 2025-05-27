"""
Document Forgery Detection System - Enhanced OCR Module
Combines Tesseract and EasyOCR for better text detection and bounding boxes
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
from typing import List, Dict, Optional, Tuple
from loguru import logger
from fuzzywuzzy import fuzz
from .models import TextDetection, BoundingBox, RotatedBoundingBox

class EnhancedOCRAnalyzer:
    """Enhanced OCR using both EasyOCR and Tesseract"""
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        """
        Initialize the enhanced OCR analyzer
        
        Args:
            languages: List of language codes for OCR
            use_gpu: Whether to use GPU acceleration for EasyOCR
        """
        self.languages = languages
        self.use_gpu = use_gpu
        
        try:
            self.easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
            logger.info(f"EasyOCR initialized with languages: {languages}, GPU: {use_gpu}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            self.easyocr_reader = None

    def extract_text_with_boxes(self, image_path: str) -> List[TextDetection]:
        """
        Extract text and bounding boxes using EasyOCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of TextDetection objects with text and bounding box information
        """
        if not self.easyocr_reader:
            logger.warning("EasyOCR not available, falling back to basic OCR")
            return self._fallback_ocr(image_path)
        
        try:
            # Read text from the image
            ocr_data = self.easyocr_reader.readtext(image_path, detail=1)
            
            results = []
            for (bbox, text, conf) in ocr_data:
                # bbox is a list of 4 points: [top_left, top_right, bottom_right, bottom_left]
                # Calculate center, width, height and angle from points
                
                # Convert polygon points to a rotated rectangle
                rect = cv2.minAreaRect(np.array(bbox, dtype=np.float32))
                # rect returns: ((center_x, center_y), (width, height), angle)
                (center_x, center_y), (width, height), angle = rect
                
                # Standard bbox coordinates for non-rotated rect (for compatibility)
                top_left = bbox[0]
                bottom_right = bbox[2]
                x = int(top_left[0])
                y = int(top_left[1])
                w = int(bottom_right[0] - top_left[0])
                h = int(bottom_right[1] - top_left[1])
                
                cleaned_text = text.strip()
                if cleaned_text:  # Ignore empty strings
                    # Create standard bounding box
                    standard_box = BoundingBox(x=x, y=y, width=w, height=h)
                    
                    # Create rotated bounding box
                    rotated_box = RotatedBoundingBox(
                        center=(int(center_x), int(center_y)),
                        size=(int(width), int(height)),
                        angle=angle,
                        points=bbox
                    )
                    
                    # Create text detection object
                    detection = TextDetection(
                        text=cleaned_text,
                        box=standard_box,
                        rotated_box=rotated_box,
                        confidence=conf
                    )
                    
                    results.append(detection)
            
            logger.info(f"EasyOCR found {len(results)} text boxes in {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error during EasyOCR processing: {e}")
            return self._fallback_ocr(image_path)

    def _fallback_ocr(self, image_path: str) -> List[TextDetection]:
        """Fallback OCR using Tesseract"""
        try:
            # Use Tesseract for basic OCR
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            
            # Basic bounding box (whole image)
            w, h = image.size
            standard_box = BoundingBox(x=0, y=0, width=w, height=h)
            
            detection = TextDetection(
                text=text.strip(),
                box=standard_box,
                rotated_box=None,
                confidence=0.5  # Default confidence for fallback
            )
            
            return [detection] if detection.text else []
            
        except Exception as e:
            logger.error(f"Fallback OCR failed: {str(e)}")
            return []

    def find_text_box(self, ocr_results: List[TextDetection], target_text: str, 
                      threshold: int = 85) -> Optional[TextDetection]:
        """
        Find the bounding box from OCR results that best matches the target text
        using fuzzy matching
        
        Args:
            ocr_results: List of OCR results
            target_text: Text to search for
            threshold: Minimum similarity threshold
            
        Returns:
            Best matching TextDetection or None
        """
        if not target_text:
            return None

        best_match = None
        highest_similarity = 0

        # Clean the target text
        cleaned_target = target_text.strip().replace(':', '').replace(',', '')

        for detection in ocr_results:
            ocr_text = detection.text
            cleaned_ocr = ocr_text.strip().replace(':', '').replace(',', '')
            
            # Calculate similarity score
            similarity_score = fuzz.WRatio(cleaned_target.lower(), cleaned_ocr.lower())

            # Track the best match above the threshold
            if similarity_score > highest_similarity and similarity_score >= threshold:
                highest_similarity = similarity_score
                best_match = detection
                
                # If perfect match, return immediately
                if similarity_score == 100:
                    break

        return best_match

    def calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: First bounding box
            box2: Second bounding box
            
        Returns:
            IoU score between 0 and 1
        """
        # Convert to (x1, y1, x2, y2) format
        box1_coords = (box1.x, box1.y, box1.x + box1.width, box1.y + box1.height)
        box2_coords = (box2.x, box2.y, box2.x + box2.width, box2.y + box2.height)
        
        # Determine intersection rectangle coordinates
        x1 = max(box1_coords[0], box2_coords[0])
        y1 = max(box1_coords[1], box2_coords[1])
        x2 = min(box1_coords[2], box2_coords[2])
        y2 = min(box1_coords[3], box2_coords[3])

        # Compute intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0

        # Compute areas of both boxes
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height

        # Compute IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def extract_document_info(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Extract basic document information using OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with document information
        """
        ocr_results = self.extract_text_with_boxes(image_path)
        
        # Combine all text for analysis
        full_text = " ".join([detection.text for detection in ocr_results])
        
        # Basic document info extraction (can be enhanced with ML models)
        info = {
            "full_text": full_text,
            "num_text_blocks": len(ocr_results),
            "avg_confidence": sum([d.confidence for d in ocr_results]) / len(ocr_results) if ocr_results else 0,
            "document_type": self._infer_document_type(full_text),
            "detected_dates": self._extract_dates(full_text),
            "detected_amounts": self._extract_amounts(full_text)
        }
        
        return info

    def _infer_document_type(self, text: str) -> Optional[str]:
        """Infer document type from text content"""
        text_lower = text.lower()
        
        document_types = {
            "invoice": ["invoice", "bill", "billing", "payment due"],
            "receipt": ["receipt", "thank you", "purchase", "transaction"],
            "bank_statement": ["bank", "statement", "balance", "account"],
            "medical_bill": ["medical", "hospital", "clinic", "patient", "diagnosis"],
            "insurance": ["insurance", "policy", "coverage", "claim"],
            "tax_document": ["tax", "irs", "w-2", "1099", "return"],
            "contract": ["contract", "agreement", "terms", "conditions"],
            "certificate": ["certificate", "certification", "award", "diploma"]
        }
        
        for doc_type, keywords in document_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        
        return "unknown"

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date patterns from text"""
        import re
        
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY or MM-DD-YYYY
            r'\d{1,2}\s+\w+\s+\d{2,4}',        # DD Month YYYY
            r'\w+\s+\d{1,2},?\s+\d{2,4}'       # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return dates

    def _extract_amounts(self, text: str) -> List[str]:
        """Extract monetary amounts from text"""
        import re
        
        amount_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',   # $1,234.56
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|CAD|EUR|GBP)',  # 1234.56 USD
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            amounts.extend(matches)
        
        return amounts

def test_enhanced_ocr():
    """Test function for the enhanced OCR analyzer"""
    analyzer = EnhancedOCRAnalyzer()
    
    # Test with a sample image (you'll need to provide a real image path)
    test_image = "test_document.jpg"
    
    try:
        results = analyzer.extract_text_with_boxes(test_image)
        print(f"Found {len(results)} text detections")
        
        for i, detection in enumerate(results):
            print(f"Detection {i+1}: '{detection.text}' at {detection.box} (conf: {detection.confidence:.2f})")
        
        # Test document info extraction
        doc_info = analyzer.extract_document_info(test_image)
        print(f"Document info: {doc_info}")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_enhanced_ocr() 