"""
Document Forgery Detection System - Fusion Module
Module 4: Heuristic ensemble for combining analysis results
"""

from typing import Dict, List, Optional
import numpy as np
from loguru import logger
import json
from datetime import datetime

class FusionAnalyzer:
    """Combines results from all analysis modules using heuristic rules"""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize with configurable thresholds
        
        Args:
            thresholds: Dict with custom threshold values
        """
        # Default thresholds - can be tuned based on validation data
        self.thresholds = {
            'suspicious_threshold': 0.45,    # Overall suspicion threshold
            'high_confidence_threshold': 0.7, # High confidence threshold
            'metadata_weight': 0.5,         # Weight for metadata analysis
            'vision_weight': 0.2,           # Weight for vision analysis
            'layout_weight': 0.3,           # Weight for layout analysis
            'ensemble_bias': 0.05           # Bias towards flagging (false positive tolerance)
        }
        
        if thresholds:
            self.thresholds.update(thresholds)
            
        logger.info(f"Fusion analyzer initialized with thresholds: {self.thresholds}")
    
    def analyze_document(self, 
                        metadata_result: Dict,
                        vision_result: Dict, 
                        layout_result: Dict,
                        document_info: Dict,
                        pixel_result: Optional[Dict] = None,
                        individual_agents_result: Optional[Dict] = None) -> Dict:
        """
        Main fusion function that combines all analysis results
        
        Args:
            metadata_result: Results from PDF metadata analysis
            vision_result: Results from vision forensics analysis
            layout_result: Results from layout reasoning analysis
            document_info: Basic document information
            pixel_result: Results from pixel-level forensics analysis (optional)
            individual_agents_result: Results from individual analysis agents (optional)
            
        Returns:
            Dict containing final analysis and recommendations
        """
        try:
            # Extract scores from each module
            scores = self._extract_scores(metadata_result, vision_result, layout_result, pixel_result, individual_agents_result)
            
            # Calculate weighted ensemble score
            ensemble_score = self._calculate_ensemble_score(scores)
            
            # Determine final verdict
            verdict = self._determine_verdict(ensemble_score, scores)
            
            # Collect all indicators
            indicators = self._collect_indicators(metadata_result, vision_result, layout_result, pixel_result, individual_agents_result)
            
            # Generate confidence assessment
            confidence_assessment = self._assess_confidence(scores, indicators)
            
            # Determine analysis modules used
            analysis_modules = ['metadata', 'vision', 'layout']
            if pixel_result and pixel_result.get('module') == 'pixel_forensics_aggregate':
                analysis_modules.append('pixel_forensics')
            if individual_agents_result and individual_agents_result.get('module') == 'individual_agents_aggregate':
                analysis_modules.append('individual_agents')
            
            # Create final report
            final_result = {
                'document_path': document_info.get('file_path', 'unknown'),
                'analysis_timestamp': datetime.now().isoformat(),
                'verdict': verdict,
                'overall_score': ensemble_score,
                'confidence': confidence_assessment['confidence'],
                'risk_level': self._determine_risk_level(ensemble_score, confidence_assessment['confidence']),
                'module_scores': scores,
                'indicators': indicators,
                'recommendations': self._generate_recommendations(verdict, ensemble_score, indicators),
                'technical_details': {
                    'processing_branch': document_info.get('processing_branch', 'unknown'),
                    'page_count': document_info.get('page_count', 0),
                    'has_text_layer': document_info.get('has_text_layer', False),
                    'file_type': document_info.get('file_type', 'unknown'),
                    'analysis_modules': analysis_modules,
                    'fusion_method': 'heuristic_ensemble_with_all_modules'
                },
                'detailed_results': {
                    'metadata_analysis': metadata_result,
                    'vision_analysis': vision_result,
                    'layout_analysis': layout_result,
                    'confidence_breakdown': confidence_assessment
                }
            }
            
            # Add pixel results if available
            if pixel_result:
                final_result['detailed_results']['pixel_analysis'] = pixel_result
            
            # Add individual agents results if available
            if individual_agents_result:
                final_result['detailed_results']['individual_agents_analysis'] = individual_agents_result
            
            logger.info(f"Document analysis complete - Verdict: {verdict}, Score: {ensemble_score:.3f}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error in fusion analysis: {str(e)}")
            return self._create_error_result(str(e), document_info)
    
    def _extract_scores(self, metadata_result: Dict, vision_result: Dict, 
                       layout_result: Dict, pixel_result: Optional[Dict] = None,
                       individual_agents_result: Optional[Dict] = None) -> Dict:
        """Extract normalized scores from each module"""
        scores = {
            'metadata': {
                'score': metadata_result.get('score', 0.0),
                'suspicious': metadata_result.get('suspicious', False),
                'available': 'error' not in metadata_result
            },
            'vision': {
                'score': vision_result.get('score', 0.0),
                'suspicious': vision_result.get('suspicious', False),
                'available': 'error' not in vision_result
            },
            'layout': {
                'score': layout_result.get('score', 0.0),
                'suspicious': layout_result.get('suspicious', False),
                'available': 'error' not in layout_result
            }
        }
        
        # Add pixel forensics scores if available
        if pixel_result and pixel_result.get('module') in ['pixel_forensics', 'pixel_forensics_aggregate']:
            scores['pixel'] = {
                'score': pixel_result.get('score', 0.0),
                'suspicious': pixel_result.get('suspicious', False),
                'available': 'error' not in pixel_result
            }
        else:
            scores['pixel'] = {
                'score': 0.0,
                'suspicious': False,
                'available': False
            }
        
        # Add individual agents scores if available
        if individual_agents_result and individual_agents_result.get('module') == 'individual_agents_aggregate':
            scores['individual_agents'] = {
                'score': individual_agents_result.get('score', 0.0),
                'suspicious': individual_agents_result.get('suspicious', False),
                'available': 'error' not in individual_agents_result
            }
        else:
            scores['individual_agents'] = {
                'score': 0.0,
                'suspicious': False,
                'available': False
            }
        
        return scores
    
    def _calculate_ensemble_score(self, scores: Dict) -> float:
        """Calculate weighted ensemble score"""
        try:
            total_weight = 0.0
            weighted_score = 0.0
            
            # Metadata component
            if scores['metadata']['available']:
                weighted_score += scores['metadata']['score'] * self.thresholds['metadata_weight']
                total_weight += self.thresholds['metadata_weight']
            
            # Vision component
            if scores['vision']['available']:
                weighted_score += scores['vision']['score'] * self.thresholds['vision_weight']
                total_weight += self.thresholds['vision_weight']
            
            # Layout component
            if scores['layout']['available']:
                weighted_score += scores['layout']['score'] * self.thresholds['layout_weight']
                total_weight += self.thresholds['layout_weight']
            
            # Pixel forensics component (add weight if available)
            if scores['pixel']['available']:
                pixel_weight = 0.25  # Weight for pixel analysis
                weighted_score += scores['pixel']['score'] * pixel_weight
                total_weight += pixel_weight
                logger.info(f"Including pixel forensics in ensemble with weight {pixel_weight}")
            
            # Individual agents component (add weight if available)
            if scores['individual_agents']['available']:
                individual_agents_weight = 0.25  # Weight for individual agents analysis
                weighted_score += scores['individual_agents']['score'] * individual_agents_weight
                total_weight += individual_agents_weight
                logger.info(f"Including individual agents analysis in ensemble with weight {individual_agents_weight}")
            
            # Normalize by available weights
            if total_weight > 0:
                ensemble_score = weighted_score / total_weight
            else:
                ensemble_score = 0.0
            
            # Apply ensemble bias (prefer false positives over false negatives)
            ensemble_score += self.thresholds['ensemble_bias']
            
            # Apply heuristic rules
            ensemble_score = self._apply_heuristic_rules(ensemble_score, scores)
            
            return min(ensemble_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating ensemble score: {str(e)}")
            return 0.5  # Default to medium suspicion on error
    
    def _apply_heuristic_rules(self, base_score: float, scores: Dict) -> float:
        """Apply heuristic rules based on module agreement"""
        
        # Rule 1: If any module flags as highly suspicious, increase score
        high_suspicion_modules = 0
        for module_name, module_data in scores.items():
            if module_data['available'] and module_data['score'] > 0.7:
                high_suspicion_modules += 1
        
        if high_suspicion_modules >= 1:
            base_score += 0.1 * high_suspicion_modules
        
        # Rule 2: If multiple modules agree on suspicion, boost confidence
        suspicious_modules = sum(1 for m in scores.values() if m['available'] and m['suspicious'])
        total_available = sum(1 for m in scores.values() if m['available'])
        
        if total_available > 0:
            agreement_ratio = suspicious_modules / total_available
            if agreement_ratio >= 0.67:  # 2/3 or more agree
                base_score += 0.15
            elif agreement_ratio >= 0.5:  # 1/2 agree
                base_score += 0.1
        
        # Rule 3: Penalize if no modules can analyze (encrypted, corrupted, etc.)
        if total_available == 0:
            base_score = 0.4  # Default suspicious for unanalyzable documents
        
        # Rule 4: Special case - metadata flags are often reliable
        if scores['metadata']['available'] and scores['metadata']['score'] > 0.5:
            base_score += 0.1
        
        # Rule 5: Pixel-level anomalies are often strong indicators
        if scores.get('pixel', {}).get('available') and scores['pixel']['score'] > 0.6:
            base_score += 0.12  # Slight bonus for pixel-level evidence
            logger.info("Applied pixel forensics bonus to ensemble score")
        
        # Rule 6: Individual agents consensus is valuable
        if scores.get('individual_agents', {}).get('available'):
            agent_score = scores['individual_agents']['score']
            if agent_score > 0.7:
                base_score += 0.15  # Strong bonus for high agent consensus
                logger.info("Applied individual agents high confidence bonus to ensemble score")
            elif agent_score > 0.5:
                base_score += 0.08  # Moderate bonus for moderate agent consensus
                logger.info("Applied individual agents moderate confidence bonus to ensemble score")
        
        # Rule 7: Arithmetic calculation errors are strong tampering indicators
        if scores['layout']['available']:
            layout_details = None
            # Try to access layout details from the calling context if available
            # This would need to be passed from the calling function
            pass  # Placeholder for now - the layout scoring already handles this
        
        return base_score
    
    def _determine_verdict(self, ensemble_score: float, scores: Dict) -> str:
        """Determine final verdict based on ensemble score and individual modules"""
        
        if ensemble_score >= self.thresholds['high_confidence_threshold']:
            return "SUSPICIOUS"
        elif ensemble_score >= self.thresholds['suspicious_threshold']:
            # Check for strong individual indicators
            strong_indicators = any(
                scores[module]['available'] and scores[module]['score'] > 0.6 
                for module in scores
            )
            return "SUSPICIOUS" if strong_indicators else "QUESTIONABLE"
        else:
            return "CLEAN"
    
    def _collect_indicators(self, metadata_result: Dict, vision_result: Dict, 
                           layout_result: Dict, pixel_result: Optional[Dict] = None,
                           individual_agents_result: Optional[Dict] = None) -> List[Dict]:
        """Collect all suspicious indicators from modules"""
        indicators = []
        
        # Metadata indicators
        if metadata_result.get('indicators'):
            for indicator in metadata_result['indicators']:
                indicators.append({
                    'source': 'metadata',
                    'description': indicator,
                    'severity': 'medium'  # Default severity
                })
        
        # Vision indicators
        if vision_result.get('indicators'):
            for indicator in vision_result['indicators']:
                indicators.append({
                    'source': 'vision',
                    'description': indicator,
                    'severity': 'high' if vision_result.get('score', 0) > 0.6 else 'medium'
                })
        
        # Layout indicators
        if layout_result.get('indicators'):
            for indicator in layout_result['indicators']:
                indicators.append({
                    'source': 'layout',
                    'description': indicator,
                    'severity': 'high' if layout_result.get('score', 0) > 0.6 else 'medium'
                })
        
        # Pixel forensics indicators
        if pixel_result and pixel_result.get('indicators'):
            for indicator in pixel_result['indicators']:
                indicators.append({
                    'source': 'pixel_forensics',
                    'description': indicator,
                    'severity': 'high' if pixel_result.get('score', 0) > 0.6 else 'medium'
                })
        
        # Individual agents indicators
        if individual_agents_result and individual_agents_result.get('indicators'):
            for indicator in individual_agents_result['indicators']:
                indicators.append({
                    'source': 'individual_agents',
                    'description': indicator,
                    'severity': 'high' if individual_agents_result.get('score', 0) > 0.6 else 'medium'
                })
        
        return indicators
    
    def _assess_confidence(self, scores: Dict, indicators: List[Dict]) -> Dict:
        """Assess confidence in the analysis"""
        
        available_modules = sum(1 for m in scores.values() if m['available'])
        total_modules = len(scores)
        
        # Base confidence from module availability
        availability_confidence = available_modules / total_modules
        
        # Confidence from indicator consistency
        if len(indicators) == 0:
            indicator_confidence = 0.9  # High confidence in clean documents
        else:
            # Lower confidence if indicators are mixed or contradictory
            sources = set(ind['source'] for ind in indicators)
            indicator_confidence = min(0.8, 0.9 - len(sources) * 0.1)
        
        # Confidence from score variance
        available_scores = [m['score'] for m in scores.values() if m['available']]
        if len(available_scores) > 1:
            score_variance = np.var(available_scores)
            variance_confidence = max(0.5, 1.0 - score_variance * 2)
        else:
            variance_confidence = 0.7  # Lower confidence with single module
        
        # Overall confidence
        overall_confidence = (
            availability_confidence * 0.4 +
            indicator_confidence * 0.3 +
            variance_confidence * 0.3
        )
        
        return {
            'confidence': min(overall_confidence, 1.0),
            'availability_factor': availability_confidence,
            'indicator_consistency': indicator_confidence,
            'score_consistency': variance_confidence,
            'available_modules': available_modules,
            'total_modules': total_modules
        }
    
    def _determine_risk_level(self, score: float, confidence: float) -> str:
        """Determine risk level based on score and confidence"""
        
        if score >= 0.7 and confidence >= 0.7:
            return "HIGH"
        elif score >= 0.5 or (score >= 0.3 and confidence >= 0.6):
            return "MEDIUM"
        elif score >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, verdict: str, score: float, indicators: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if verdict == "SUSPICIOUS":
            recommendations.append("⚠️  Document flagged as SUSPICIOUS - Recommend manual review")
            recommendations.append("🔍 Detailed forensic analysis recommended")
            
            if any(ind['source'] == 'metadata' for ind in indicators):
                recommendations.append("📄 Review document creation and modification history")
            
            if any(ind['source'] == 'vision' for ind in indicators):
                recommendations.append("👁️  Visual inspection for copy-paste or digital alterations")
            
            if any(ind['source'] == 'layout' for ind in indicators):
                recommendations.append("🧮 Verify all arithmetic calculations manually")
            
            if any(ind['source'] == 'pixel_forensics' for ind in indicators):
                recommendations.append("🔬 Pixel-level anomalies detected - Check for digital manipulation")
            
            if any(ind['source'] == 'individual_agents' for ind in indicators):
                recommendations.append("🤖 Specialized agents flagged concerns - Detailed content review needed")
        
        elif verdict == "QUESTIONABLE":
            recommendations.append("❓ Document has questionable elements - Review recommended")
            recommendations.append("📋 Cross-reference with original sources if available")
        
        elif verdict == "CLEAN":
            recommendations.append("✅ No obvious signs of tampering detected")
            if score > 0.1:
                recommendations.append("ℹ️  Minor inconsistencies noted but within normal range")
        
        # General recommendations
        if len(indicators) > 0:
            high_severity = sum(1 for ind in indicators if ind.get('severity') == 'high')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-priority issue(s) require attention")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, document_info: Dict) -> Dict:
        """Create error result when fusion fails"""
        return {
            'document_path': document_info.get('file_path', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat(),
            'verdict': "ERROR",
            'overall_score': 0.5,  # Default to suspicious on error
            'confidence': 0.0,
            'risk_level': "UNKNOWN",
            'error': error_message,
            'indicators': [{'source': 'system', 'description': f"Analysis error: {error_message}", 'severity': 'high'}],
            'recommendations': [
                "❌ Analysis failed - Manual review required",
                "🔧 Check document format and integrity",
                "📞 Contact technical support if issue persists"
            ],
            'technical_details': document_info
        }


def test_fusion_analyzer():
    """Test function for the fusion analyzer"""
    analyzer = FusionAnalyzer()
    
    # Mock test data
    mock_metadata = {'score': 0.2, 'suspicious': False, 'indicators': []}
    mock_vision = {'score': 0.8, 'suspicious': True, 'indicators': ['Inconsistent font weights']}
    mock_layout = {'score': 0.1, 'suspicious': False, 'indicators': []}
    mock_doc_info = {'file_path': 'test.pdf', 'page_count': 1}
    
    result = analyzer.analyze_document(mock_metadata, mock_vision, mock_layout, mock_doc_info)
    
    logger.info(f"Test result: {result['verdict']} (Score: {result['overall_score']:.3f})")


if __name__ == "__main__":
    test_fusion_analyzer() 