"""
Document Forgery Detection System - Report Generation Module
Creates comprehensive analysis reports with visual overlays
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from loguru import logger

class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Report generator initialized - Output directory: {output_dir}")
    
    def generate_report(self, analysis_result: Dict, save_files: bool = True) -> Dict:
        """
        Generate comprehensive analysis report
        
        Args:
            analysis_result: Complete analysis results from fusion module
            save_files: Whether to save report files to disk
            
        Returns:
            Dict containing report data and file paths
        """
        try:
            # Extract basic info
            document_path = analysis_result.get('document_path', 'unknown')
            timestamp = analysis_result.get('analysis_timestamp', datetime.now().isoformat())
            
            # Generate report ID
            report_id = self._generate_report_id(document_path, timestamp)
            
            # Create report structure
            report = {
                'report_id': report_id,
                'timestamp': timestamp,
                'document_info': {
                    'path': document_path,
                    'filename': os.path.basename(document_path),
                    'technical_details': analysis_result.get('technical_details', {})
                },
                'analysis_summary': self._create_analysis_summary(analysis_result),
                'detailed_findings': self._create_detailed_findings(analysis_result),
                'visual_evidence': self._create_visual_evidence(analysis_result),
                'recommendations': analysis_result.get('recommendations', []),
                'technical_report': self._create_technical_report(analysis_result)
            }
            
            # Save files if requested
            saved_files = {}
            if save_files:
                saved_files = self._save_report_files(report, report_id, analysis_result)
                report['saved_files'] = saved_files
            
            logger.info(f"Report generated successfully: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return self._create_error_report(str(e), analysis_result)
    
    def _generate_report_id(self, document_path: str, timestamp: str) -> str:
        """Generate unique report ID"""
        import hashlib
        
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(document_path))[0]
        
        # Create hash from filename and timestamp
        hash_input = f"{filename}_{timestamp}".encode()
        short_hash = hashlib.md5(hash_input).hexdigest()[:8]
        
        # Format: FILENAME_YYYYMMDD_HASH
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{filename}_{date_str}_{short_hash}"
    
    def _create_analysis_summary(self, result: Dict) -> Dict:
        """Create high-level analysis summary"""
        return {
            'verdict': result.get('verdict', 'UNKNOWN'),
            'overall_score': result.get('overall_score', 0.0),
            'confidence': result.get('confidence', 0.0),
            'risk_level': result.get('risk_level', 'UNKNOWN'),
            'total_indicators': len(result.get('indicators', [])),
            'module_results': {
                module: {
                    'score': data.get('score', 0.0),
                    'suspicious': data.get('suspicious', False),
                    'available': data.get('available', False)
                }
                for module, data in result.get('module_scores', {}).items()
            }
        }
    
    def _create_detailed_findings(self, result: Dict) -> Dict:
        """Create detailed findings breakdown"""
        indicators = result.get('indicators', [])
        
        # Group indicators by source and severity
        grouped_indicators = {
            'metadata': [],
            'vision': [],
            'layout': [],
            'system': []
        }
        
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for indicator in indicators:
            source = indicator.get('source', 'unknown')
            severity = indicator.get('severity', 'medium')
            
            if source in grouped_indicators:
                grouped_indicators[source].append(indicator)
            
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return {
            'indicators_by_source': grouped_indicators,
            'severity_breakdown': severity_counts,
            'key_findings': self._extract_key_findings(indicators),
            'module_details': result.get('detailed_results', {})
        }
    
    def _extract_key_findings(self, indicators: List[Dict]) -> List[str]:
        """Extract most important findings"""
        key_findings = []
        
        # High-severity findings
        high_severity = [ind for ind in indicators if ind.get('severity') == 'high']
        for finding in high_severity[:3]:  # Top 3 high-severity
            key_findings.append(f"🚨 {finding.get('description', 'Unknown issue')}")
        
        # Important metadata findings
        metadata_findings = [ind for ind in indicators if ind.get('source') == 'metadata']
        for finding in metadata_findings[:2]:  # Top 2 metadata
            key_findings.append(f"📄 {finding.get('description', 'Unknown metadata issue')}")
        
        # Visual findings
        vision_findings = [ind for ind in indicators if ind.get('source') == 'vision']
        for finding in vision_findings[:2]:  # Top 2 vision
            key_findings.append(f"👁️ {finding.get('description', 'Unknown visual issue')}")
        
        return key_findings[:5]  # Limit to top 5 overall
    
    def _create_visual_evidence(self, result: Dict) -> Dict:
        """Create visual evidence section"""
        visual_evidence = {
            'page_analyses': [],
            'overlay_maps': [],
            'suspicious_regions': []
        }
        
        # Extract vision analysis results
        vision_details = result.get('detailed_results', {}).get('vision_analysis', {})
        
        if 'page_results' in vision_details:
            for page_result in vision_details['page_results']:
                page_info = {
                    'page_number': page_result.get('page_number', 1),
                    'suspicious': page_result.get('suspicious', False),
                    'confidence': page_result.get('confidence', 0.0),
                    'suspected_regions': page_result.get('suspected_regions', [])
                }
                visual_evidence['page_analyses'].append(page_info)
                
                # Add suspicious regions
                for region in page_result.get('suspected_regions', []):
                    visual_evidence['suspicious_regions'].append({
                        'page': page_result.get('page_number', 1),
                        'description': region.get('description', 'Unknown region'),
                        'confidence': region.get('confidence', 0.0)
                    })
        
        return visual_evidence
    
    def _create_technical_report(self, result: Dict) -> Dict:
        """Create detailed technical report"""
        return {
            'analysis_modules': result.get('technical_details', {}).get('analysis_modules', []),
            'processing_details': {
                'branch': result.get('technical_details', {}).get('processing_branch', 'unknown'),
                'page_count': result.get('technical_details', {}).get('page_count', 0),
                'text_layer': result.get('technical_details', {}).get('has_text_layer', False),
                'file_type': result.get('technical_details', {}).get('file_type', 'unknown')
            },
            'module_performance': self._analyze_module_performance(result),
            'confidence_breakdown': result.get('detailed_results', {}).get('confidence_breakdown', {}),
            'raw_scores': result.get('module_scores', {})
        }
    
    def _analyze_module_performance(self, result: Dict) -> Dict:
        """Analyze performance of each module"""
        module_scores = result.get('module_scores', {})
        
        performance = {}
        for module, data in module_scores.items():
            performance[module] = {
                'operational': data.get('available', False),
                'score': data.get('score', 0.0),
                'contribution': 'high' if data.get('score', 0) > 0.5 else 'low',
                'reliability': 'high' if data.get('available', False) else 'low'
            }
        
        return performance
    
    def _create_visual_overlays(self, analysis_result: Dict, report_id: str) -> List[str]:
        """Create visual overlays showing suspicious regions"""
        overlay_files = []
        
        try:
            # Get vision analysis details
            vision_details = analysis_result.get('detailed_results', {}).get('vision_analysis', {})
            
            if 'page_results' in vision_details:
                for page_result in vision_details['page_results']:
                    if page_result.get('suspicious', False):
                        overlay_file = self._create_page_overlay(page_result, report_id)
                        if overlay_file:
                            overlay_files.append(overlay_file)
        
        except Exception as e:
            logger.warning(f"Could not create visual overlays: {str(e)}")
        
        return overlay_files
    
    def _create_page_overlay(self, page_result: Dict, report_id: str) -> Optional[str]:
        """Create overlay for a single page"""
        try:
            page_number = page_result.get('page_number', 1)
            image_path = page_result.get('details', {}).get('image_analyzed', '')
            
            if not image_path or not os.path.exists(image_path):
                return None
            
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Create overlay
            overlay = image.copy()
            
            # Add suspicious region highlights (simplified - no bounding boxes available)
            if page_result.get('suspicious', False):
                # Add a red tint to indicate suspicion
                red_overlay = np.zeros_like(overlay)
                red_overlay[:, :] = [0, 0, 255]  # Red in BGR
                
                # Blend with original
                overlay = cv2.addWeighted(overlay, 0.9, red_overlay, 0.1, 0)
                
                # Add text indicating suspicion
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"SUSPICIOUS CONTENT DETECTED (Page {page_number})"
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = (overlay.shape[1] - text_size[0]) // 2
                text_y = 50
                
                # Add text with background
                cv2.rectangle(overlay, (text_x-10, text_y-30), (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
                cv2.putText(overlay, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
            
            # Save overlay
            overlay_filename = f"{report_id}_page_{page_number}_overlay.png"
            overlay_path = os.path.join(self.output_dir, overlay_filename)
            cv2.imwrite(overlay_path, overlay)
            
            return overlay_path
            
        except Exception as e:
            logger.warning(f"Could not create page overlay: {str(e)}")
            return None
    
    def _save_report_files(self, report: Dict, report_id: str, analysis_result: Dict) -> Dict:
        """Save all report files to disk"""
        saved_files = {}
        
        try:
            # Save JSON report
            json_filename = f"{report_id}_report.json"
            json_path = os.path.join(self.output_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            saved_files['json_report'] = json_path
            
            # Save human-readable summary
            summary_filename = f"{report_id}_summary.txt"
            summary_path = os.path.join(self.output_dir, summary_filename)
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self._create_text_summary(report))
            
            saved_files['text_summary'] = summary_path
            
            # Create visual overlays
            overlay_files = self._create_visual_overlays(analysis_result, report_id)
            if overlay_files:
                saved_files['visual_overlays'] = overlay_files
            
            logger.info(f"Report files saved: {list(saved_files.keys())}")
            
        except Exception as e:
            logger.error(f"Error saving report files: {str(e)}")
        
        return saved_files
    
    def _create_text_summary(self, report: Dict) -> str:
        """Create human-readable text summary"""
        summary = []
        
        # Header
        summary.append("=" * 80)
        summary.append("DOCUMENT FORGERY DETECTION REPORT")
        summary.append("=" * 80)
        summary.append("")
        
        # Basic info
        summary.append(f"Report ID: {report['report_id']}")
        summary.append(f"Timestamp: {report['timestamp']}")
        summary.append(f"Document: {report['document_info']['filename']}")
        summary.append("")
        
        # Analysis summary
        analysis = report['analysis_summary']
        summary.append("ANALYSIS SUMMARY")
        summary.append("-" * 40)
        summary.append(f"Verdict: {analysis['verdict']}")
        summary.append(f"Overall Score: {analysis['overall_score']:.3f}")
        summary.append(f"Confidence: {analysis['confidence']:.3f}")
        summary.append(f"Risk Level: {analysis['risk_level']}")
        summary.append(f"Total Indicators: {analysis['total_indicators']}")
        summary.append("")
        
        # Key findings
        if report['detailed_findings']['key_findings']:
            summary.append("KEY FINDINGS")
            summary.append("-" * 40)
            for finding in report['detailed_findings']['key_findings']:
                summary.append(f"• {finding}")
            summary.append("")
        
        # Recommendations
        if report['recommendations']:
            summary.append("RECOMMENDATIONS")
            summary.append("-" * 40)
            for rec in report['recommendations']:
                summary.append(f"• {rec}")
            summary.append("")
        
        # Module results
        summary.append("MODULE RESULTS")
        summary.append("-" * 40)
        for module, data in analysis['module_results'].items():
            status = "✓" if data['available'] else "✗"
            summary.append(f"{status} {module.upper()}: Score {data['score']:.3f} {'(SUSPICIOUS)' if data['suspicious'] else '(CLEAN)'}")
        summary.append("")
        
        # Footer
        summary.append("=" * 80)
        summary.append("End of Report")
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def _create_error_report(self, error_message: str, partial_result: Dict) -> Dict:
        """Create error report when generation fails"""
        return {
            'report_id': f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'error': error_message,
            'document_info': {
                'path': partial_result.get('document_path', 'unknown'),
                'filename': 'unknown'
            },
            'analysis_summary': {
                'verdict': 'ERROR',
                'overall_score': 0.0,
                'confidence': 0.0,
                'risk_level': 'UNKNOWN'
            },
            'recommendations': ['❌ Report generation failed - Manual review required']
        }


def test_report_generator():
    """Test function for the report generator"""
    generator = ReportGenerator()
    
    # Mock analysis result
    mock_result = {
        'document_path': 'test_document.pdf',
        'verdict': 'SUSPICIOUS',
        'overall_score': 0.75,
        'confidence': 0.8,
        'risk_level': 'HIGH',
        'indicators': [
            {'source': 'vision', 'description': 'Inconsistent font weights', 'severity': 'high'},
            {'source': 'metadata', 'description': 'Missing document ID', 'severity': 'medium'}
        ],
        'recommendations': ['Manual review recommended'],
        'module_scores': {
            'metadata': {'score': 0.4, 'suspicious': False, 'available': True},
            'vision': {'score': 0.8, 'suspicious': True, 'available': True},
            'layout': {'score': 0.2, 'suspicious': False, 'available': True}
        },
        'technical_details': {'page_count': 1, 'file_type': 'pdf'}
    }
    
    report = generator.generate_report(mock_result, save_files=False)
    logger.info(f"Test report generated: {report['report_id']}")


if __name__ == "__main__":
    test_report_generator() 