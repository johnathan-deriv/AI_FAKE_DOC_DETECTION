"""
Document Forgery Detection System - Vision Forensics Module
Module 2B: LLM-based visual sanity checks using OpenAI GPT models
"""

import base64
import json
from typing import Dict, List, Optional, Tuple
from PIL import Image
import io
import openai
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

class VisionForensicsAnalyzer:
    """Uses OpenAI vision APIs for zero-shot tampering detection"""
    
    def __init__(self, model_type: str = "reasoning"):
        """
        Initialize with OpenAI model preference
        
        Args:
            model_type: "reasoning" for o4-mini or "standard" for gpt-4.1
        """
        self.model_type = model_type
        self.setup_apis()
        
        # Prompts for different types of analysis
        self.visual_forensics_prompt = """You are a forensic analyst specializing in document tampering detection. 
Inspect this business quotation/invoice page carefully for ANY visible signs of digital manipulation.

Look specifically for:
1. Copy-paste blocks (repeated textures, misaligned edges)
2. Different font weights or styles within similar text sections
3. Inconsistent shadows, lighting, or image quality
4. Smudged, blurred, or artificially sharp areas
5. Misaligned stamps, signatures, or logos
6. Inconsistent background patterns or textures
7. JPEG compression artifacts that don't match the rest
8. Color or brightness inconsistencies

If you find ANY suspicious visual artifacts, respond with JSON:
{
  "tampered": "yes",
  "confidence": 0.7,
  "reasons": ["specific reason 1", "specific reason 2"],
  "suspected_regions": [
    {"description": "top right corner numbers", "confidence": 0.8},
    {"description": "signature area", "confidence": 0.6}
  ]
}

If everything looks natural and consistent, respond with:
{
  "tampered": "no",
  "confidence": 0.9,
  "reasons": [],
  "suspected_regions": []
}

Only respond with valid JSON. Be very specific about what you observe."""

        self.quality_check_prompt = """Analyze this document image for quality issues that might indicate tampering:

1. Resolution inconsistencies across different areas
2. Compression artifacts (JPEG blocking) that vary across the image
3. Noise patterns that don't match
4. Focus/blur inconsistencies
5. Color space or saturation differences

Respond with JSON format focusing on technical image quality indicators."""

    def setup_apis(self):
        """Setup OpenAI API clients"""
        try:
            # OpenAI setup
            openai_key = os.getenv('OPENAI_API_KEY')
            
            # Get model names from environment variables
            self.reasoning_model_name = os.getenv('OPENAI_MODEL_NAME1', 'o3')  # Default to o3 if not set
            self.standard_model_name = os.getenv('OPENAI_MODEL_NAME2', 'gpt-4.1')   # Default to gpt-4.1 if not set
            
            # Set current model and temperature based on model_type
            if self.model_type == "reasoning":
                self.current_model_name = self.reasoning_model_name
                self.temperature = 1  # Reasoning models only support temperature=1
            else:
                self.current_model_name = self.standard_model_name
                self.temperature = 0  # Standard models use temperature=0 for deterministic results
            
            if openai_key:
                openai.api_key = openai_key
                self.openai_client = openai.OpenAI(api_key=openai_key)
                logger.info(f"OpenAI API initialized with model: {self.current_model_name}, temperature: {self.temperature}")
            else:
                self.openai_client = None
                logger.error("OpenAI API key not found")
                raise Exception("OpenAI API key is required")
                
        except Exception as e:
            logger.error(f"Error setting up OpenAI API: {str(e)}")
            self.openai_client = None
            raise e

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image for visual tampering signs
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing analysis results
        """
        try:
            if not self.openai_client:
                raise Exception("OpenAI API not available")

            # Load and prepare image
            with Image.open(image_path) as img:
                # Resize if too large (for API limits)
                if max(img.size) > 2048:
                    img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Analyze with OpenAI
                result = self._analyze_with_openai(img)
                
                # Post-process results
                return self._process_vision_results(result, image_path)
                
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            return {
                'module': 'vision_forensics',
                'score': 0.0,
                'suspicious': False,
                'error': str(e),
                'indicators': [f"Could not analyze image: {str(e)}"],
                'details': {}
            }

    def _analyze_with_openai(self, image: Image.Image) -> Dict:
        """Analyze using OpenAI GPT models"""
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            response = self.openai_client.chat.completions.create(
                model=self.current_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.visual_forensics_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=1000,
                temperature=self.temperature
            )
            
            result_text = response.choices[0].message.content
            logger.info(f"OpenAI response: {result_text}")
            
            # Parse JSON response
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise Exception(f"Could not parse JSON from response: {result_text}")
                    
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise e

    def _process_vision_results(self, api_result: Dict, image_path: str) -> Dict:
        """Process and standardize API results"""
        try:
            tampered = api_result.get('tampered', 'no').lower() == 'yes'
            confidence = api_result.get('confidence', 0.5)
            reasons = api_result.get('reasons', [])
            suspected_regions = api_result.get('suspected_regions', [])
            
            # Calculate suspicion score based on confidence and findings
            if tampered:
                score = confidence * 0.8  # Base score from confidence
                score += min(len(reasons) * 0.1, 0.2)  # Bonus for multiple reasons
                score = min(score, 1.0)
            else:
                score = (1 - confidence) * 0.3  # Low score for clean documents
            
            return {
                'module': 'vision_forensics',
                'model_used': self.current_model_name,
                'model_type': self.model_type,
                'score': score,
                'suspicious': tampered,
                'confidence': confidence,
                'indicators': reasons,
                'suspected_regions': suspected_regions,
                'details': {
                    'image_analyzed': image_path,
                    'tampered_detected': tampered,
                    'region_count': len(suspected_regions),
                    'raw_response': api_result
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing vision results: {str(e)}")
            return {
                'module': 'vision_forensics',
                'score': 0.3,  # Default suspicious score for processing errors
                'suspicious': True,
                'error': str(e),
                'indicators': [f"Could not process vision analysis: {str(e)}"],
                'details': {}
            }

    def analyze_multiple_pages(self, page_data: List[Dict]) -> Dict:
        """
        Analyze multiple pages and aggregate results
        
        Args:
            page_data: List of page dictionaries with image_path
            
        Returns:
            Aggregated analysis results
        """
        page_results = []
        total_score = 0.0
        all_indicators = []
        all_regions = []
        
        for page in page_data:
            if 'image_path' in page:
                result = self.analyze_image(page['image_path'])
                result['page_number'] = page.get('page_number', 1)
                page_results.append(result)
                
                total_score += result.get('score', 0.0)
                all_indicators.extend(result.get('indicators', []))
                all_regions.extend(result.get('suspected_regions', []))
        
        # Calculate aggregate score
        avg_score = total_score / len(page_results) if page_results else 0.0
        
        return {
            'module': 'vision_forensics_aggregate',
            'model_used': self.current_model_name,
            'model_type': self.model_type,
            'score': avg_score,
            'suspicious': avg_score > 0.4,
            'page_count': len(page_results),
            'indicators': all_indicators,
            'suspected_regions': all_regions,
            'page_results': page_results,
            'details': {
                'total_suspicious_pages': sum(1 for r in page_results if r.get('suspicious', False)),
                'average_confidence': sum(r.get('confidence', 0) for r in page_results) / len(page_results) if page_results else 0
            }
        }


def test_vision_analyzer():
    """Test function for the vision forensics analyzer"""
    # Test with reasoning model
    print("=== Testing Vision Forensics Analyzer with Reasoning Model ===")
    analyzer_reasoning = VisionForensicsAnalyzer(model_type="reasoning")
    logger.info(f"Vision forensics analyzer initialized with reasoning model: {analyzer_reasoning.current_model_name}, temperature: {analyzer_reasoning.temperature}")
    
    # Test with standard model
    print("\n=== Testing Vision Forensics Analyzer with Standard Model ===")
    analyzer_standard = VisionForensicsAnalyzer(model_type="standard")
    logger.info(f"Vision forensics analyzer initialized with standard model: {analyzer_standard.current_model_name}, temperature: {analyzer_standard.temperature}")
    
    print("\n=== Vision Model Configuration Summary ===")
    print(f"Reasoning Model: {analyzer_reasoning.reasoning_model_name} (temperature: {analyzer_reasoning.temperature})")
    print(f"Standard Model: {analyzer_standard.standard_model_name} (temperature: {analyzer_standard.temperature})")
    print("Note: Reasoning models (o1/o4) only support temperature=1")
    print("      Standard models use temperature=0 for deterministic results")
    
    # Test if APIs are available
    if analyzer_reasoning.openai_client and analyzer_standard.openai_client:
        print("\n✅ Both model types are ready for vision analysis")
    else:
        print("\n⚠️ Check OpenAI API key configuration")


if __name__ == "__main__":
    test_vision_analyzer() 