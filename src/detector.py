"""
Document Forgery Detection System - Main Detector
Orchestrates all analysis modules for the no-training approach
"""

import os
import tempfile
from typing import Dict, List, Optional, Union
from loguru import logger
import numpy as np
from datetime import datetime

from .ingest import DocumentIngestor
from .pdf_meta import PDFMetadataAnalyzer
from .vision_net import VisionForensicsAnalyzer
from .pixel_forensics import PixelForensicsAnalyzer
from .layout_reasoner import LayoutReasoningAnalyzer
from .individual_agents import IndividualAnalysisAgents
from .fusion import FusionAnalyzer
from .report import ReportGenerator

class DocumentForgeryDetector:
    """
    Main detector class that orchestrates the complete analysis pipeline
    Based on the no-data, no-training approach using LLM APIs
    """
    
    def __init__(self, 
                 vision_model_type: str = "standard",
                 text_model_type: str = "reasoning",
                 output_dir: str = "reports",
                 fusion_thresholds: Optional[Dict] = None,
                 enable_pixel_analysis: bool = True,
                 enable_individual_agents: bool = False,
                 pixel_block_size: int = 16,
                 fast_pixel_mode: bool = True,
                 enhanced_pixel_forensics: Union[bool, str] = False):
        """
        Initialize the document forgery detector
        
        Args:
            vision_model_type: "reasoning" for o4-mini or "standard" for gpt-4.1 for vision analysis
            text_model_type: "reasoning" for o4-mini or "standard" for gpt-4.1 for text analysis
            output_dir: Directory for saving reports
            fusion_thresholds: Custom thresholds for fusion analysis
            enable_pixel_analysis: Whether to run pixel-level forensics
            enable_individual_agents: Whether to run specialized individual agents
            pixel_block_size: Block size for pixel analysis (default 16x16 for speed)
            fast_pixel_mode: Enable fast mode optimizations for pixel analysis
            enhanced_pixel_forensics: Enhanced pixel forensics mode:
                                     - False/disabled: No enhanced features
                                     - True/basic_fast: Standard enhanced features
                                     - "enhanced_fast": Optimized enhanced mode (recommended)
                                     - "enhanced_thorough": Full analysis (most comprehensive)
        """
        self.vision_model_type = vision_model_type
        self.text_model_type = text_model_type
        self.enable_pixel_analysis = enable_pixel_analysis
        self.enable_individual_agents = enable_individual_agents
        self.enhanced_pixel_forensics = enhanced_pixel_forensics
        
        # Initialize all modules
        self.ingestor = DocumentIngestor()
        self.pdf_analyzer = PDFMetadataAnalyzer()
        self.vision_analyzer = VisionForensicsAnalyzer(model_type=vision_model_type)
        self.layout_analyzer = LayoutReasoningAnalyzer(model_type=text_model_type)
        self.fusion_analyzer = FusionAnalyzer(thresholds=fusion_thresholds)
        self.report_generator = ReportGenerator(output_dir=output_dir)
        
        # Initialize pixel-level forensics analyzer if enabled
        if enable_pixel_analysis:
            self.pixel_analyzer = PixelForensicsAnalyzer(
                block_size=pixel_block_size, 
                fast_mode=fast_pixel_mode,
                enhanced_mode=enhanced_pixel_forensics
            )
            
            forensics_type = "Enhanced" if enhanced_pixel_forensics else "Standard"
            logger.info(f"{forensics_type} pixel-level forensics enabled with block size {pixel_block_size}x{pixel_block_size}, fast_mode={fast_pixel_mode}")
            
            if enhanced_pixel_forensics:
                logger.info("Advanced techniques enabled: DCT analysis, wavelet decomposition, copy-move detection, frequency domain analysis")
        else:
            self.pixel_analyzer = None
            logger.info("Pixel-level forensics disabled")
        
        # Initialize individual analysis agents if enabled
        if enable_individual_agents:
            self.individual_agents = IndividualAnalysisAgents(model_type=text_model_type)
            logger.info("Individual analysis agents enabled")
        else:
            self.individual_agents = None
            logger.info("Individual analysis agents disabled")
        
        logger.info("Document Forgery Detector initialized successfully")
        logger.info(f"Vision Model: {vision_model_type}, Text Model: {text_model_type}")
        logger.info(f"Optional modules: Pixel={enable_pixel_analysis} (Enhanced={enhanced_pixel_forensics}), Individual Agents={enable_individual_agents}")
    
    def analyze_document(self, file_path: str, save_report: bool = True, max_pages: int = 3) -> Dict:
        """
        Main analysis function following the implementation sketch from the markdown
        
        Implementation follows this flow:
        1. save original file → tmp
        2. pdf_meta_check() → rule_score
        3. for each page:
              img = pdf2image(page)
              vis_json = call_gpt4v(img, VIS_PROMPT)
              pixel_json = pixel_level_analysis(img)  # NEW
              text = pytesseract(img)
              logic_json = call_gpt35(text, ARITH_PROMPT)
              suspicious |= (vis_json['tampered'] or pixel_json['suspicious'] or logic_json['mismatch'])
        4. if suspicious or rule_score>0 → produce report.pdf with overlay boxes
        
        Args:
            file_path: Path to the document to analyze
            save_report: Whether to save detailed report files
            max_pages: Maximum number of pages to analyze
            
        Returns:
            Dict containing complete analysis results
        """
        logger.info(f"Starting analysis of document: {file_path}")
        
        try:
            # Step 1: Document ingestion and preprocessing
            logger.info("Step 1: Document ingestion and preprocessing")
            document_info = self.ingestor.process_document(file_path)
            page_data = document_info.get('pages', [])
            
            if not page_data:
                logger.warning("No pages extracted from document")
                return self._create_error_result("No pages could be extracted", file_path)
            
            # Step 2: PDF metadata analysis (if applicable)
            metadata_result = {'module': 'metadata', 'score': 0.0, 'suspicious': False, 'indicators': []}
            if document_info.get('file_type') == 'pdf':
                logger.info("Step 2: PDF metadata analysis")
                try:
                    # Get metadata from document_info
                    pdf_metadata = document_info.get('metadata', {})
                    metadata_result = self.pdf_analyzer.analyze_metadata(file_path, pdf_metadata)
                    logger.info(f"PDF metadata rule score: {metadata_result.get('score', 0.0):.3f}")
                except Exception as e:
                    logger.warning(f"Metadata analysis failed: {str(e)}")
                    metadata_result['error'] = str(e)
            else:
                logger.info("Step 2: Metadata analysis skipped (not a PDF)")
            
            # Step 3: Processing pages with vision, pixel, and text analysis
            logger.info("Step 3: Processing pages with vision, pixel, and text analysis")
            
            # Vision analysis (Module 2V)
            logger.info("Step 3a: Vision forensics analysis")
            try:
                vision_result = self.vision_analyzer.analyze_multiple_pages(page_data)
            except Exception as e:
                logger.warning(f"Vision analysis failed: {str(e)}")
                vision_result = {
                    'module': 'vision_forensics',
                    'score': 0.0,
                    'suspicious': False,
                    'error': str(e),
                    'indicators': [f"Vision analysis failed: {str(e)}"]
                }
            
            # Pixel-level analysis (Module 2C)
            pixel_result = {'module': 'pixel_forensics', 'score': 0.0, 'suspicious': False, 'indicators': []}
            if self.enable_pixel_analysis and self.pixel_analyzer:
                logger.info("Step 3b: Pixel-level forensics analysis")
                try:
                    pixel_result = self._analyze_pixel_forensics(page_data)
                except Exception as e:
                    logger.warning(f"Pixel analysis failed: {str(e)}")
                    pixel_result = {
                        'module': 'pixel_forensics',
                        'score': 0.0,
                        'suspicious': False,
                        'error': str(e),
                        'indicators': [f"Pixel analysis failed: {str(e)}"]
                    }
            else:
                logger.info("Step 3b: Pixel-level analysis skipped (disabled)")
            
            # Text/Layout analysis (Module 3)
            logger.info("Step 3c: Text and layout reasoning")
            try:
                layout_result = self.layout_analyzer.analyze_document_logic(page_data)
            except Exception as e:
                logger.warning(f"Layout analysis failed: {str(e)}")
                layout_result = {
                    'module': 'layout_reasoning',
                    'score': 0.0,
                    'suspicious': False,
                    'error': str(e),
                    'indicators': [f"Layout analysis failed: {str(e)}"]
                }

            # Individual Agents analysis (Module 4)
            individual_agents_result = {'module': 'individual_agents', 'score': 0.0, 'suspicious': False, 'indicators': []}
            if self.enable_individual_agents and self.individual_agents:
                logger.info("Step 3d: Individual agents analysis")
                try:
                    individual_agents_result = self._analyze_individual_agents(file_path, max_pages)
                except Exception as e:
                    logger.warning(f"Individual agents analysis failed: {str(e)}")
                    individual_agents_result = {
                        'module': 'individual_agents',
                        'score': 0.0,
                        'suspicious': False,
                        'error': str(e),
                        'indicators': [f"Individual agents analysis failed: {str(e)}"]
                    }
            else:
                logger.info("Step 3d: Individual agents analysis skipped (disabled)")
            
            # Step 4: Fusion analysis and final verdict
            logger.info("Step 4: Fusion analysis and final verdict")
            
            # Check individual module suspicion flags for logging
            metadata_suspicious = metadata_result.get('suspicious', False)
            vision_suspicious = vision_result.get('suspicious', False)
            pixel_suspicious = pixel_result.get('suspicious', False)
            layout_suspicious = layout_result.get('suspicious', False)
            individual_agents_suspicious = individual_agents_result.get('suspicious', False)
            rule_score = metadata_result.get('score', 0.0)
            
            suspicious = metadata_suspicious or vision_suspicious or pixel_suspicious or layout_suspicious or individual_agents_suspicious
            
            logger.info(f"Individual module suspicion flags: metadata={metadata_suspicious}, vision={vision_suspicious}, pixel={pixel_suspicious}, layout={layout_suspicious}, individual_agents={individual_agents_suspicious}, rule_score={rule_score}")
            
            # Perform fusion analysis
            fusion_result = self.fusion_analyzer.analyze_document(
                metadata_result, vision_result, layout_result, document_info, pixel_result, individual_agents_result
            )
            
            logger.info(f"Fusion analysis complete - Verdict: {fusion_result.get('verdict', 'unknown')}, Score: {fusion_result.get('overall_score', 0.0):.3f}")
            
            # Step 5: Create visualizations if needed
            annotated_images = None
            if save_report and (suspicious or fusion_result.get('overall_score', 0.0) > 0.1):
                logger.info("Step 5a: Creating visual overlays")
                try:
                    annotated_images = self._create_visual_overlays(pixel_result, vision_result, page_data)
                    if annotated_images:
                        logger.info(f"Created {len(annotated_images)} annotated images")
                except Exception as e:
                    logger.warning(f"Visualization creation failed: {str(e)}")
            
            # Step 5b: Generate report if suspicious or requested
            if save_report and (suspicious or fusion_result.get('overall_score', 0.0) > 0.1):
                logger.info("Step 5b: Generating detailed report")
                try:
                    # Add annotated images to fusion result before report generation
                    if annotated_images:
                        fusion_result['annotated_images'] = annotated_images
                    
                    report = self.report_generator.generate_report(fusion_result, save_files=True)
                    fusion_result['report'] = report
                    
                    # Add annotated images to report for web display
                    if annotated_images:
                        fusion_result['report']['annotated_images'] = annotated_images
                        
                except Exception as e:
                    logger.warning(f"Report generation failed: {str(e)}")
                    fusion_result['report_error'] = str(e)
            
            # Cleanup temporary files
            self.ingestor.cleanup()
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            # Cleanup on error
            try:
                self.ingestor.cleanup()
            except:
                pass
            
            return self._create_error_result(str(e), file_path)
    
    def _analyze_pixel_forensics(self, page_data: List[Dict]) -> Dict:
        """
        Analyze multiple pages using pixel-level forensics and aggregate results
        
        Args:
            page_data: List of page dictionaries with image_path
            
        Returns:
            Aggregated pixel forensics results
        """
        page_results = []
        total_score = 0.0
        all_indicators = []
        all_anomaly_regions = []
        
        for page in page_data:
            if 'image_path' in page:
                result = self.pixel_analyzer.analyze_image(page['image_path'])
                result['page_number'] = page.get('page_number', 1)
                page_results.append(result)
                
                total_score += result.get('score', 0.0)
                all_indicators.extend(result.get('indicators', []))
                all_anomaly_regions.extend(result.get('anomaly_regions', []))
        
        # Calculate aggregate score
        avg_score = total_score / len(page_results) if page_results else 0.0
        
        return {
            'module': 'pixel_forensics_aggregate',
            'score': avg_score,
            'suspicious': avg_score > 0.4,
            'page_count': len(page_results),
            'indicators': all_indicators,
            'anomaly_regions': all_anomaly_regions,
            'page_results': page_results,
            'details': {
                'total_suspicious_pages': sum(1 for r in page_results if r.get('suspicious', False)),
                'total_anomaly_regions': len(all_anomaly_regions),
                'average_confidence': sum(r.get('score', 0) for r in page_results) / len(page_results) if page_results else 0,
                'pixel_analysis_enabled': True
            }
        }

    def _analyze_individual_agents(self, file_path: str, max_pages: int = 3) -> Dict:
        """
        Run individual analysis agents and aggregate their results
        
        Args:
            file_path: Path to the document file
            max_pages: Maximum number of pages to analyze
            
        Returns:
            Aggregated individual agents results
        """
        try:
            # Run all individual agents
            agents_results = self.individual_agents.run_all_agents(file_path, number_of_pages=max_pages)
            
            # Extract summary information
            summary = agents_results.get('_summary', {})
            
            # Get individual agent results (excluding summary)
            individual_results = {k: v for k, v in agents_results.items() if k != '_summary'}
            
            # Aggregate indicators from all agents
            all_indicators = []
            total_score = 0.0
            suspicious_agents = []
            
            for agent_name, result in individual_results.items():
                all_indicators.extend(result.get('indicators', []))
                total_score += result.get('score', 0.0)
                
                if result.get('suspicious', False):
                    suspicious_agents.append(agent_name)
            
            # Calculate average score
            avg_score = total_score / len(individual_results) if individual_results else 0.0
            
            return {
                'module': 'individual_agents_aggregate',
                'score': avg_score,
                'suspicious': summary.get('suspicious', False),
                'indicators': list(set(all_indicators)),  # Remove duplicates
                'agent_results': individual_results,
                'summary': summary,
                'details': {
                    'total_agents': len(individual_results),
                    'suspicious_agents': suspicious_agents,
                    'suspicious_agent_count': len(suspicious_agents),
                    'average_confidence': avg_score,
                    'agents_run': list(individual_results.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Error in individual agents analysis: {str(e)}")
            return {
                'module': 'individual_agents_aggregate',
                'score': 0.0,
                'suspicious': False,
                'indicators': [f"Individual agents analysis failed: {str(e)}"],
                'error': str(e),
                'details': {}
            }
    
    def analyze_batch(self, file_paths: List[str], save_reports: bool = True) -> Dict:
        """
        Analyze multiple documents in batch
        
        Args:
            file_paths: List of document paths to analyze
            save_reports: Whether to save detailed reports
            
        Returns:
            Dict containing batch analysis results
        """
        logger.info(f"Starting batch analysis of {len(file_paths)} documents")
        
        batch_results = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'total_documents': len(file_paths),
            'completed': 0,
            'suspicious_count': 0,
            'clean_count': 0,
            'error_count': 0,
            'results': []
        }
        
        for i, file_path in enumerate(file_paths, 1):
            logger.info(f"Processing document {i}/{len(file_paths)}: {file_path}")
            
            try:
                result = self.analyze_document(file_path, save_report=save_reports)
                
                batch_results['results'].append({
                    'file_path': file_path,
                    'result': result
                })
                
                # Update counters
                verdict = result.get('verdict', 'ERROR')
                if verdict == 'SUSPICIOUS':
                    batch_results['suspicious_count'] += 1
                elif verdict == 'CLEAN':
                    batch_results['clean_count'] += 1
                else:
                    batch_results['error_count'] += 1
                
                batch_results['completed'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                batch_results['results'].append({
                    'file_path': file_path,
                    'error': str(e)
                })
                batch_results['error_count'] += 1
        
        logger.info(f"Batch analysis complete - {batch_results['suspicious_count']} suspicious, "
                   f"{batch_results['clean_count']} clean, {batch_results['error_count']} errors")
        
        return batch_results
    
    def _create_error_result(self, error_message: str, file_path: str) -> Dict:
        """Create standardized error result"""
        return {
            'document_path': file_path,
            'analysis_timestamp': datetime.now().isoformat(),
            'verdict': 'ERROR',
            'overall_score': 0.5,  # Default to suspicious on error
            'confidence': 0.0,
            'risk_level': 'UNKNOWN',
            'error': error_message,
            'indicators': [{'source': 'system', 'description': f"Analysis error: {error_message}", 'severity': 'high'}],
            'recommendations': [
                "❌ Analysis failed - Manual review required",
                "🔧 Check document format and integrity",
                "📞 Contact technical support if issue persists"
            ],
            'module_scores': {
                'metadata': {'score': 0.0, 'suspicious': False, 'available': False},
                'vision': {'score': 0.0, 'suspicious': False, 'available': False},
                'layout': {'score': 0.0, 'suspicious': False, 'available': False}
            },
            'technical_details': {
                'processing_branch': 'unknown',
                'page_count': 0,
                'has_text_layer': False,
                'file_type': 'unknown',
                'analysis_modules': ['metadata', 'vision', 'layout'],
                'fusion_method': 'heuristic_ensemble'
            }
        }
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status and component availability"""
        import pytesseract
        import cv2
        from PIL import Image
        
        status = {
            'detector_ready': True,
            'timestamp': datetime.now().isoformat(),
            
            # API Status
            'apis': {
                'openai_vision': bool(self.vision_analyzer.openai_client),
                'openai_text': bool(self.layout_analyzer.openai_client)
            },
            
            # Core Analysis Modules Status
            'core_modules': {
                'pdf_ingestor': True,  # Always available (built-in)
                'pdf_metadata_analyzer': True,  # Always available (built-in)
                'vision_forensics_analyzer': bool(self.vision_analyzer.openai_client),
                'layout_reasoning_analyzer': bool(self.layout_analyzer.openai_client),
                'fusion_analyzer': True,  # Always available (rule-based)
                'report_generator': True  # Always available (built-in)
            },
            
            # Optional Enhancement Modules Status
            'optional_modules': {
                'pixel_forensics_analyzer': bool(self.pixel_analyzer),
                'individual_agents_analyzer': bool(self.individual_agents),
                'enhanced_ocr_system': self._check_enhanced_ocr_availability(),
                'visualization_engine': self._check_visualization_availability()
            },
            
            # System Dependencies Status
            'dependencies': {
                'tesseract_ocr': self._check_tesseract_availability(),
                'opencv': self._check_opencv_availability(),
                'pillow_imaging': self._check_pillow_availability(),
                'pdf_processing': self._check_pdf_processing_availability(),
                'numpy_computing': self._check_numpy_availability()
            },
            
            # File System & Resources
            'system_resources': {
                'output_directory': os.path.exists(self.report_generator.output_dir) and os.access(self.report_generator.output_dir, os.W_OK),
                'temp_directory': os.access(os.path.expanduser('~'), os.W_OK),
                'memory_status': 'available',  # Could be enhanced with actual memory check
                'storage_status': 'available'   # Could be enhanced with actual storage check
            },
            
            # Configuration Status
            'configuration': {
                'vision_model_type': self.vision_model_type,
                'text_model_type': self.text_model_type,
                'vision_model_name': getattr(self.vision_analyzer, 'current_model_name', 'unknown'),
                'text_model_name': getattr(self.layout_analyzer, 'current_model_name', 'unknown'),
                'vision_temperature': getattr(self.vision_analyzer, 'temperature', 0),
                'text_temperature': getattr(self.layout_analyzer, 'temperature', 0),
                'pixel_analysis_enabled': bool(self.pixel_analyzer),
                'individual_agents_enabled': bool(self.individual_agents),
                'fast_pixel_mode': self.pixel_analyzer.fast_mode if self.pixel_analyzer else False,
                'pixel_block_size': self.pixel_analyzer.block_size if self.pixel_analyzer else None
            }
        }
        
        # Calculate overall system health
        api_health = status['apis']['openai_vision'] and status['apis']['openai_text']
        
        core_modules_health = all(status['core_modules'].values())
        dependencies_health = all(status['dependencies'].values())
        resources_health = all(status['system_resources'].values())
        
        # Overall system status
        status['system_health'] = {
            'apis_operational': api_health,
            'core_modules_operational': core_modules_health,
            'dependencies_operational': dependencies_health,
            'resources_operational': resources_health,
            'overall_operational': api_health and core_modules_health and dependencies_health and resources_health
        }
        
        # Legacy compatibility (for existing code)
        status['modules'] = status['core_modules']
        status['vision_api_available'] = status['apis']['openai_vision']
        status['text_api_available'] = status['apis']['openai_text']
        status['system_operational'] = status['system_health']['overall_operational']
        
        return status
    
    def _check_enhanced_ocr_availability(self) -> bool:
        """Check if enhanced OCR system is available"""
        try:
            # Check if EasyOCR is available
            import easyocr
            return True
        except ImportError:
            return False
    
    def _check_visualization_availability(self) -> bool:
        """Check if visualization engine is available"""
        try:
            import cv2
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            return False
    
    def _check_tesseract_availability(self) -> bool:
        """Check if Tesseract OCR is available"""
        try:
            import pytesseract
            # Try to get version to ensure it's properly installed
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _check_opencv_availability(self) -> bool:
        """Check if OpenCV is available"""
        try:
            import cv2
            return hasattr(cv2, '__version__')
        except ImportError:
            return False
    
    def _check_pillow_availability(self) -> bool:
        """Check if Pillow (PIL) is available"""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def _check_pdf_processing_availability(self) -> bool:
        """Check if PDF processing libraries are available"""
        try:
            import PyPDF2
            from pdf2image import convert_from_path
            return True
        except ImportError:
            return False
    
    def _check_numpy_availability(self) -> bool:
        """Check if NumPy is available"""
        try:
            import numpy as np
            return True
        except ImportError:
            return False

    def _create_visual_overlays(self, pixel_result: Dict, vision_result: Dict, page_data: List[Dict]) -> List[np.ndarray]:
        """
        Create visual overlays showing detected anomaly regions and suspicious areas
        
        Args:
            pixel_result: Results from pixel forensics analysis
            vision_result: Results from vision analysis
            page_data: List of page information with image paths
            
        Returns:
            List of annotated images as numpy arrays
        """
        try:
            # Import visualization after we're sure we need it
            from .visualization import DocumentVisualizer
            from .models import ForgeryIndicator, BoundingBox
            import cv2
            
            visualizer = DocumentVisualizer()
            annotated_images = []
            
            # Get pixel anomaly regions - these are per-page results
            pixel_page_results = pixel_result.get('page_results', [])
            vision_page_results = vision_result.get('page_results', [])
            
            for page_info in page_data:
                page_number = page_info.get('page_number', 1)
                image_path = page_info.get('image_path')
                
                if not image_path or not os.path.exists(image_path):
                    logger.warning(f"Image path not found for page {page_number}: {image_path}")
                    continue
                
                # Load base image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                overlay_applied = False
                
                # Find corresponding pixel analysis results for this page
                pixel_page_result = None
                for pr in pixel_page_results:
                    if pr.get('page_number') == page_number:
                        pixel_page_result = pr
                        break
                
                # Apply pixel forensics overlays
                if pixel_page_result and pixel_page_result.get('anomaly_regions'):
                    anomaly_regions = pixel_page_result['anomaly_regions']
                    logger.info(f"Drawing {len(anomaly_regions)} pixel anomaly regions on page {page_number}")
                    
                    for region in anomaly_regions:
                        # Draw bounding box around anomaly region
                        x, y, w, h = region['x'], region['y'], region['width'], region['height']
                        confidence = region.get('confidence', 0.0)
                        anomaly_type = region.get('type', 'unknown')
                        
                        # Color based on anomaly type and confidence
                        if confidence > 0.7:
                            color = (0, 0, 255)  # Red for high confidence
                        elif confidence > 0.5:
                            color = (0, 165, 255)  # Orange for medium confidence
                        else:
                            color = (0, 255, 255)  # Yellow for lower confidence
                        
                        # Draw rectangle
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                        
                        # Add label
                        label = f"{anomaly_type[:10]}:{confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # Background for text
                        cv2.rectangle(image, (x, y - label_size[1] - 5), 
                                    (x + label_size[0] + 5, y), color, -1)
                        # Text
                        cv2.putText(image, label, (x + 2, y - 2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        overlay_applied = True
                
                # Find corresponding vision analysis results for this page
                vision_page_result = None
                for vr in vision_page_results:
                    if vr.get('page_number') == page_number:
                        vision_page_result = vr
                        break
                
                # Apply vision analysis overlays (if any suspicious regions detected)
                if vision_page_result and vision_page_result.get('suspicious', False):
                    suspected_regions = vision_page_result.get('suspected_regions', [])
                    
                    if suspected_regions:
                        logger.info(f"Drawing {len(suspected_regions)} vision suspicious regions on page {page_number}")
                        
                        for region in suspected_regions:
                            # If the region has coordinates, draw them
                            if 'bbox' in region:
                                bbox = region['bbox']
                                x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('width', 0), bbox.get('height', 0)
                                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # Blue for vision
                                
                                label = f"Vision: {region.get('description', 'suspicious')}"
                                cv2.putText(image, label, (x, y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                overlay_applied = True
                    
                    # If no specific regions but page is suspicious, add general indicator
                    elif not overlay_applied:
                        # Add border and text to indicate suspicious content
                        h, w = image.shape[:2]
                        cv2.rectangle(image, (5, 5), (w-5, h-5), (255, 0, 0), 8)  # Blue border
                        
                        # Add text at top
                        text = f"SUSPICIOUS CONTENT DETECTED (Page {page_number})"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        text_x = (w - text_size[0]) // 2
                        text_y = 50
                        
                        # Text background
                        cv2.rectangle(image, (text_x-10, text_y-30), 
                                    (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
                        # Text
                        cv2.putText(image, text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        overlay_applied = True
                
                # If no overlays were applied but we have suspicious indicators, add generic highlighting
                if not overlay_applied:
                    # Check if this page has any indicators from any module
                    has_indicators = (
                        (pixel_page_result and (
                            pixel_page_result.get('suspicious', False) or 
                            pixel_page_result.get('score', 0) > 0.3
                        )) or
                        (vision_page_result and vision_page_result.get('suspicious', False))
                    )
                    
                    if has_indicators:
                        # Add a subtle highlight border
                        h, w = image.shape[:2]
                        cv2.rectangle(image, (2, 2), (w-2, h-2), (0, 255, 255), 4)  # Yellow border
                        
                        # Add indicator text
                        text = f"Potential Issues Detected (Page {page_number})"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                        text_x = (w - text_size[0]) // 2
                        text_y = 30
                        
                        cv2.rectangle(image, (text_x-5, text_y-20), 
                                    (text_x+text_size[0]+5, text_y+5), (0, 0, 0), -1)
                        cv2.putText(image, text, (text_x, text_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                annotated_images.append(image)
                logger.info(f"Created annotated image for page {page_number}")
            
            return annotated_images
            
        except Exception as e:
            logger.error(f"Error creating visual overlays: {str(e)}")
            return []


def test_detector():
    """Test function for the main detector"""
    # Test with reasoning models (default)
    print("=== Testing Document Detector with Reasoning Models ===")
    detector_reasoning = DocumentForgeryDetector(
        vision_model_type="reasoning",
        text_model_type="reasoning"
    )
    
    status = detector_reasoning.get_system_status()
    logger.info(f"Reasoning model system status: {status}")
    
    print("\n=== Testing Document Detector with Standard Models ===")
    detector_standard = DocumentForgeryDetector(
        vision_model_type="standard",
        text_model_type="standard"
    )
    
    status_standard = detector_standard.get_system_status()
    logger.info(f"Standard model system status: {status_standard}")
    
    # Test mixed configuration
    print("\n=== Testing Document Detector with Mixed Models ===")
    detector_mixed = DocumentForgeryDetector(
        vision_model_type="standard",   # Fast for vision
        text_model_type="reasoning"     # Reasoning for complex text analysis
    )
    
    status_mixed = detector_mixed.get_system_status()
    
    print("\n=== Model Configuration Summary ===")
    config_reasoning = status.get("configuration", {})
    config_standard = status_standard.get("configuration", {})
    config_mixed = status_mixed.get("configuration", {})
    
    print(f"Reasoning Configuration:")
    print(f"  Vision: {config_reasoning.get('vision_model_name')} (temp: {config_reasoning.get('vision_temperature')})")
    print(f"  Text: {config_reasoning.get('text_model_name')} (temp: {config_reasoning.get('text_temperature')})")
    
    print(f"\nStandard Configuration:")
    print(f"  Vision: {config_standard.get('vision_model_name')} (temp: {config_standard.get('vision_temperature')})")
    print(f"  Text: {config_standard.get('text_model_name')} (temp: {config_standard.get('text_temperature')})")
    
    print(f"\nMixed Configuration:")
    print(f"  Vision: {config_mixed.get('vision_model_name')} (temp: {config_mixed.get('vision_temperature')})")
    print(f"  Text: {config_mixed.get('text_model_name')} (temp: {config_mixed.get('text_temperature')})")
    
    print("\nNote: Reasoning models (o1/o4) use temperature=1")
    print("      Standard models use temperature=0 for deterministic results")
    
    if status['system_operational']:
        logger.info("✅ Document Forgery Detector is ready for operation")
    else:
        logger.warning("⚠️ Some APIs are not available - check configuration")


if __name__ == "__main__":
    test_detector() 