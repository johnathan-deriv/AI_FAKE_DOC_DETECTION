"""
Document Forgery Detection System - Data Models
Pydantic models for structured data handling and API responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class OCRInfo(BaseModel):
    """Information extracted from a document via OCR"""
    document_type: Optional[str] = Field(default=None, description="The type of document (e.g., bank statement, medical bill, invoice, receipt)")
    organization: Optional[str] = Field(default=None, description="The name of the organization that issued the document")
    country: Optional[str] = Field(default=None, description="The country where the document was issued")
    customer_name: Optional[str] = Field(default=None, description="The name of the customer/patient/recipient")

class BoundingBox(BaseModel):
    """Bounding box coordinates for detected text/regions"""
    x: int = Field(description="X coordinate of top-left corner")
    y: int = Field(description="Y coordinate of top-left corner") 
    width: int = Field(description="Width of the bounding box")
    height: int = Field(description="Height of the bounding box")

class RotatedBoundingBox(BaseModel):
    """Rotated bounding box for text detection"""
    center: tuple[int, int] = Field(description="Center coordinates (x, y)")
    size: tuple[int, int] = Field(description="Size (width, height)")
    angle: float = Field(description="Rotation angle in degrees")
    points: List[List[float]] = Field(description="Four corner points")

class TextDetection(BaseModel):
    """OCR text detection result"""
    text: str = Field(description="Detected text content")
    box: BoundingBox = Field(description="Standard bounding box")
    rotated_box: Optional[RotatedBoundingBox] = Field(default=None, description="Rotated bounding box if applicable")
    confidence: float = Field(description="Detection confidence score")

class ForgeryIndicator(BaseModel):
    """Individual forgery indicator result"""
    indicator_type: str = Field(description="Type of forgery indicator")
    text: Optional[str] = Field(default=None, description="Relevant text from the document")
    box: Optional[BoundingBox] = Field(default=None, description="Bounding box for visualization")
    confidence: float = Field(description="Confidence score for this indicator")
    description: str = Field(description="Human-readable description")

class IndividualAnalysisResult(BaseModel):
    """Result from an individual analysis agent"""
    legitimacy: str = Field(description="Assessment: 'fake' or 'legit'")
    reasoning: str = Field(description="Detailed reasoning for the assessment")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    indicators: Optional[List[ForgeryIndicator]] = Field(default=None, description="Specific forgery indicators found")

class FinalAnalysisResult(BaseModel):
    """Final comprehensive analysis result"""
    legitimacy: str = Field(description="Final assessment: 'fake' or 'legit'")
    reasoning: str = Field(description="Comprehensive reasoning for the assessment")
    forgery_indicators: Optional[List[str]] = Field(default=None, description="List of forgery indicator types found")
    overall_confidence: float = Field(description="Overall confidence score")
    individual_results: Optional[Dict[str, IndividualAnalysisResult]] = Field(default=None, description="Results from individual analysis modules")

class VisionAnalysisResult(BaseModel):
    """Vision analysis result structure"""
    tampered: str = Field(description="'yes' or 'no' for tampering detection")
    confidence: float = Field(description="Confidence score")
    reasons: List[str] = Field(description="Specific reasons for the assessment")
    suspected_regions: List[Dict[str, Any]] = Field(description="Suspicious regions with descriptions")

class DocumentClassificationResult(BaseModel):
    """Complete document classification result"""
    fake_doc: bool = Field(description="Whether document is classified as fake")
    legitimacy_reasoning: str = Field(description="Reasoning for the classification")
    fake_indicator: Optional[List[str]] = Field(default=None, description="Types of forgery indicators found")
    bounding_boxes: Optional[List[ForgeryIndicator]] = Field(default=None, description="Bounding boxes for visualization")
    annotated_image: Optional[Any] = Field(default=None, description="Annotated image with indicators")
    annotated_images: Optional[List[Any]] = Field(default=None, description="Annotated images for multi-page documents")
    single_page: bool = Field(default=True, description="Whether document is single page")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score to reference documents")
    similar_document_name: Optional[str] = Field(default=None, description="Name of similar reference document")
    similar_document_legitimacy: Optional[bool] = Field(default=None, description="Legitimacy of similar reference document")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")

class APIErrorResponse(BaseModel):
    """Error response for API endpoints"""
    error: str = Field(description="Error message")
    details: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the error occurred")

class APISuccessResponse(BaseModel):
    """Success response wrapper for API endpoints"""
    success: bool = Field(default=True, description="Success status")
    data: DocumentClassificationResult = Field(description="The analysis results")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated") 