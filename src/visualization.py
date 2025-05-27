"""
Document Forgery Detection System - Visualization Module
Advanced visualization capabilities for forgery indicators with bounding boxes
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from PIL import Image
from loguru import logger

from .models import ForgeryIndicator, BoundingBox, RotatedBoundingBox, TextDetection
from .enhanced_ocr import EnhancedOCRAnalyzer

class DocumentVisualizer:
    """Advanced document visualization with forgery indicator overlays"""
    
    def __init__(self):
        """Initialize the document visualizer"""
        self.color_map = {
            'customer_name': (0, 0, 255),         # Red
            'digital_artifacts': (0, 255, 0),     # Lime Green
            'font_inconsistencies': (255, 0, 0),  # Blue
            'layout_issues': (0, 255, 255),       # Yellow
            'design_verification': (255, 0, 255), # Magenta
            'content_analysis': (0, 165, 255),    # Orange
            'date_verification': (128, 0, 128),   # Purple
            'default': (128, 128, 128)            # Gray
        }
        
        self.ocr_analyzer = EnhancedOCRAnalyzer()

    def draw_forgery_indicators(self, image_path: str, indicators: List[ForgeryIndicator], 
                              overlap_threshold: float = 0.8) -> Optional[np.ndarray]:
        """
        Draw forgery indicators on the document image
        
        Args:
            image_path: Path to the document image
            indicators: List of forgery indicators to draw
            overlap_threshold: Threshold for overlap detection to avoid duplicate boxes
            
        Returns:
            Annotated image as numpy array or None if failed
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Error loading image for drawing: {image_path}")
                return None
            
            logger.info(f"Drawing {len(indicators)} forgery indicators on {image_path}")
            
            # Get OCR results for text matching
            ocr_results = self.ocr_analyzer.extract_text_with_boxes(image_path)
            
            # Track drawn box coordinates to avoid overlaps
            drawn_boxes_coords = []
            
            # Process each indicator
            for indicator in indicators:
                try:
                    # Get color for this indicator type
                    color = self.color_map.get(indicator.indicator_type.lower(), self.color_map['default'])
                    
                    # Find the corresponding text box if we have specific text
                    if indicator.text and not indicator.box:
                        # Find matching text box from OCR results
                        matching_detection = self.ocr_analyzer.find_text_box(ocr_results, indicator.text)
                        if matching_detection:
                            indicator.box = matching_detection.box
                            logger.info(f"Found matching text box for '{indicator.text}'")
                    
                    # Draw the indicator
                    if indicator.box:
                        self._draw_indicator_box(image, indicator, color, drawn_boxes_coords, overlap_threshold)
                    else:
                        logger.warning(f"No bounding box found for indicator: {indicator.indicator_type}")
                        
                except Exception as e:
                    logger.error(f"Error drawing indicator {indicator.indicator_type}: {str(e)}")
                    continue
            
            logger.info(f"Successfully drew {len(drawn_boxes_coords)} indicator boxes")
            return image
            
        except Exception as e:
            logger.error(f"Error in draw_forgery_indicators: {str(e)}")
            return None

    def _draw_indicator_box(self, image: np.ndarray, indicator: ForgeryIndicator, 
                          color: Tuple[int, int, int], drawn_boxes: List, 
                          overlap_threshold: float):
        """
        Draw a single indicator box on the image
        
        Args:
            image: The image to draw on
            indicator: The forgery indicator to draw
            color: BGR color tuple for the box
            drawn_boxes: List to track drawn boxes for overlap detection
            overlap_threshold: Threshold for overlap detection
        """
        try:
            box = indicator.box
            
            # Check if we have rotated box information
            if hasattr(box, 'rotated_box') and box.rotated_box and hasattr(box.rotated_box, 'points'):
                # Draw rotated box using contours
                self._draw_rotated_box(image, box.rotated_box, color)
                drawn_boxes.append(True)  # Mark as drawn
                logger.debug(f"Drew rotated box for {indicator.indicator_type}")
                
            else:
                # Draw standard rectangle
                x, y, w, h = box.x, box.y, box.width, box.height
                current_box_coords = (x, y, x + w, y + h)
                
                # Check for overlap with existing boxes
                is_overlapping = False
                for drawn_coords in drawn_boxes:
                    if isinstance(drawn_coords, tuple):  # Skip non-coordinate entries
                        if self._calculate_box_iou(current_box_coords, drawn_coords) > overlap_threshold:
                            logger.debug(f"Skipping box for '{indicator.text or indicator.indicator_type}' due to overlap")
                            is_overlapping = True
                            break
                
                if not is_overlapping:
                    # Draw the rectangle
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                    drawn_boxes.append(current_box_coords)
                    logger.debug(f"Drew standard box for {indicator.indicator_type}")
                    
        except Exception as e:
            logger.error(f"Error drawing indicator box: {str(e)}")

    def _draw_rotated_box(self, image: np.ndarray, rotated_box: RotatedBoundingBox, 
                         color: Tuple[int, int, int]):
        """
        Draw a rotated bounding box using contours
        
        Args:
            image: The image to draw on
            rotated_box: The rotated bounding box information
            color: BGR color tuple for the box
        """
        try:
            points = rotated_box.points
            
            # Convert points to proper array format for contours
            box_points = []
            for point in points:
                if hasattr(point, '__iter__'):  # Check if it's iterable
                    box_points.append([int(float(point[0])), int(float(point[1]))])
                else:
                    logger.warning(f"Unexpected point format: {point}, type: {type(point)}")
                    continue
            
            # Draw the contour if we have valid points
            if box_points and len(box_points) >= 3:  # Need at least 3 points for a polygon
                box_points_array = np.array(box_points, dtype=np.int32)
                cv2.drawContours(image, [box_points_array], 0, color, 3)
                logger.debug(f"Drew rotated box with {len(box_points)} points")
            else:
                logger.warning(f"Not enough valid points for rotated box: {box_points}")
                
        except Exception as e:
            logger.error(f"Error drawing rotated box: {str(e)}")

    def _calculate_box_iou(self, box1: Tuple[int, int, int, int], 
                          box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: (x1, y1, x2, y2) coordinates
            box2: (x1, y1, x2, y2) coordinates
            
        Returns:
            IoU score between 0 and 1
        """
        # Determine intersection rectangle coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        if inter_area == 0:
            return 0

        # Compute areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute IoU
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def create_legend_html(self, detected_indicator_types: set) -> str:
        """
        Create HTML legend for detected indicator types
        
        Args:
            detected_indicator_types: Set of indicator types that were detected
            
        Returns:
            HTML string for the legend
        """
        if not detected_indicator_types:
            return ""

        # Map keys to display names and colors
        display_map = {
            'customer_name': ("Customer Name", '#FF0000'),
            'digital_artifacts': ("Digital Artifacts", '#00FF00'),
            'font_inconsistencies': ("Font Inconsistencies", '#0000FF'),
            'layout_issues': ("Layout Issues", '#FFFF00'),
            'design_verification': ("Design Verification", '#FF00FF'),
            'content_analysis': ("Content Analysis", '#FFA500'),
            'date_verification': ("Date Verification", '#800080'),
        }
        
        legend_html = "<div style='display: flex; flex-wrap: wrap;'>"
        for indicator_type in sorted(list(detected_indicator_types)):
            display_name, color_hex = display_map.get(
                indicator_type.lower(), 
                ("Unknown Indicator", '#808080')
            )

            legend_html += f"""
            <div style="margin-right: 20px; margin-bottom: 10px; display: flex; align-items: center;">
                <span style="background-color:{color_hex}; border-radius: 3px; width: 18px; height: 18px; display: inline-block; margin-right: 8px; border: 1px solid #ccc;"></span>
                <span>{display_name}</span>
            </div>
            """
        legend_html += "</div>"
        
        return legend_html

    def create_simple_indicator_overlay(self, image_path: str, indicator_text: str) -> Optional[np.ndarray]:
        """
        Create a simple colored border overlay for documents without specific bounding boxes
        
        Args:
            image_path: Path to the document image
            indicator_text: The forgery indicator text
            
        Returns:
            Image with colored border or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Error loading image: {image_path}")
                return None
            
            # Determine border color based on indicator type
            border_color = (0, 0, 255)  # Default red
            
            if isinstance(indicator_text, str):
                if "digital_artifacts" in indicator_text.lower():
                    border_color = (0, 255, 0)  # Green
                elif "font" in indicator_text.lower():
                    border_color = (255, 0, 0)  # Blue
                elif "layout" in indicator_text.lower():
                    border_color = (0, 255, 255)  # Yellow
                elif "design" in indicator_text.lower():
                    border_color = (255, 0, 255)  # Magenta
                elif "content" in indicator_text.lower():
                    border_color = (0, 165, 255)  # Orange
                elif "date" in indicator_text.lower():
                    border_color = (128, 0, 128)  # Purple
            
            # Add thick border
            border_thickness = 10
            height, width = image.shape[:2]
            
            # Draw border
            cv2.rectangle(image, (0, 0), (width-1, height-1), border_color, border_thickness)
            
            return image
            
        except Exception as e:
            logger.error(f"Error creating simple indicator overlay: {str(e)}")
            return None

    def convert_to_rgb(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to RGB for display in web interfaces
        
        Args:
            bgr_image: Image in BGR format (OpenCV default)
            
        Returns:
            Image in RGB format
        """
        try:
            return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error converting BGR to RGB: {str(e)}")
            return bgr_image

    def save_annotated_image(self, annotated_image: np.ndarray, output_path: str) -> bool:
        """
        Save annotated image to file
        
        Args:
            annotated_image: The annotated image array
            output_path: Path where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = cv2.imwrite(output_path, annotated_image)
            if success:
                logger.info(f"Annotated image saved to {output_path}")
            else:
                logger.error(f"Failed to save annotated image to {output_path}")
            return success
        except Exception as e:
            logger.error(f"Error saving annotated image: {str(e)}")
            return False

    def create_multi_page_annotations(self, page_images: List[str], 
                                    indicators_per_page: List[List[ForgeryIndicator]]) -> List[np.ndarray]:
        """
        Create annotations for multiple pages
        
        Args:
            page_images: List of page image paths
            indicators_per_page: List of indicator lists for each page
            
        Returns:
            List of annotated images
        """
        annotated_images = []
        
        for i, (page_path, indicators) in enumerate(zip(page_images, indicators_per_page)):
            try:
                logger.info(f"Annotating page {i+1} with {len(indicators)} indicators")
                annotated_image = self.draw_forgery_indicators(page_path, indicators)
                
                if annotated_image is not None:
                    annotated_images.append(annotated_image)
                else:
                    logger.warning(f"Failed to annotate page {i+1}")
                    
            except Exception as e:
                logger.error(f"Error annotating page {i+1}: {str(e)}")
                continue
        
        logger.info(f"Successfully annotated {len(annotated_images)} out of {len(page_images)} pages")
        return annotated_images

def test_visualizer():
    """Test function for the document visualizer"""
    visualizer = DocumentVisualizer()
    
    # Create test indicators
    test_indicators = [
        ForgeryIndicator(
            indicator_type="digital_artifacts",
            text="Test Company Inc.",
            box=BoundingBox(x=100, y=100, width=200, height=50),
            confidence=0.8,
            description="Suspicious logo quality"
        ),
        ForgeryIndicator(
            indicator_type="font_inconsistencies", 
            text="$1,234.56",
            box=BoundingBox(x=300, y=200, width=100, height=30),
            confidence=0.9,
            description="Font mismatch in amount"
        )
    ]
    
    # Test with a sample image (you'll need to provide a real image)
    test_image = "test_document.jpg"
    
    try:
        annotated_image = visualizer.draw_forgery_indicators(test_image, test_indicators)
        
        if annotated_image is not None:
            print("Visualization test successful")
            
            # Test legend creation
            detected_types = {indicator.indicator_type for indicator in test_indicators}
            legend_html = visualizer.create_legend_html(detected_types)
            print(f"Legend HTML: {legend_html[:100]}...")
            
            # Test saving
            success = visualizer.save_annotated_image(annotated_image, "test_output.jpg")
            print(f"Save test: {'Success' if success else 'Failed'}")
            
        else:
            print("Visualization test failed - no annotated image returned")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_visualizer() 