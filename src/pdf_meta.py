"""
Document Forgery Detection System - PDF Metadata Analysis Module
Module 2A: PDF metadata & incremental-save forensic analysis
"""

import pikepdf
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import re

class PDFMetadataAnalyzer:
    """Analyzes PDF metadata for signs of tampering"""
    
    def __init__(self):
        self.suspicious_indicators = []
        self.score = 0.0
        
    def analyze_metadata(self, file_path: str, metadata: Dict) -> Dict:
        """
        Main analysis function for PDF metadata
        
        Args:
            file_path: Path to PDF file
            metadata: Pre-extracted metadata from ingest module
            
        Returns:
            Dict containing analysis results and suspicion score
        """
        self.suspicious_indicators = []
        self.score = 0.0
        
        try:
            # Perform detailed analysis using pikepdf
            with pikepdf.open(file_path) as pdf:
                
                # Check incremental saves
                self._check_incremental_saves(pdf)
                
                # Check date consistency
                self._check_date_consistency(pdf, metadata)
                
                # Check producer/creator consistency
                self._check_producer_consistency(metadata)
                
                # Check document ID presence
                self._check_document_id(metadata)
                
                # Check object structure
                self._check_object_structure(pdf)
                
                # Check for encryption flags
                self._check_encryption_flags(pdf, metadata)
                
        except Exception as e:
            logger.error(f"Error analyzing PDF metadata: {str(e)}")
            self.suspicious_indicators.append(f"Could not analyze PDF structure: {str(e)}")
            self.score += 0.3
        
        return {
            'module': 'pdf_metadata',
            'score': min(self.score, 1.0),  # Cap at 1.0
            'suspicious': self.score > 0.3,
            'indicators': self.suspicious_indicators,
            'details': {
                'has_incremental_saves': any('incremental' in ind.lower() for ind in self.suspicious_indicators),
                'date_inconsistency': any('date' in ind.lower() for ind in self.suspicious_indicators),
                'missing_document_id': any('document id' in ind.lower() for ind in self.suspicious_indicators),
                'producer_mismatch': any('producer' in ind.lower() for ind in self.suspicious_indicators)
            }
        }
    
    def _check_incremental_saves(self, pdf: pikepdf.Pdf):
        """Check for suspicious incremental saves"""
        try:
            trailer = pdf.trailer
            
            if '/Prev' in trailer:
                prev_offset = int(trailer['/Prev'])
                logger.info(f"Found incremental save with /Prev offset: {prev_offset}")
                
                # Check if there are multiple incremental saves
                xref_count = 1  # Current xref
                current_trailer = trailer
                
                while '/Prev' in current_trailer:
                    xref_count += 1
                    # This is simplified - in a real implementation you'd parse the xref chain
                    break  # Prevent infinite loop for now
                
                if xref_count > 1:
                    self.suspicious_indicators.append(
                        f"Document has {xref_count} incremental saves - may indicate post-creation modifications"
                    )
                    self.score += 0.4
                    
                    # Check if the last revision is unsigned (simplified check)
                    if not self._has_digital_signature(pdf):
                        self.suspicious_indicators.append(
                            "Last revision appears to be unsigned after incremental save"
                        )
                        self.score += 0.5
                        
        except Exception as e:
            logger.warning(f"Could not check incremental saves: {str(e)}")
    
    def _check_date_consistency(self, pdf: pikepdf.Pdf, metadata: Dict):
        """Check for date inconsistencies"""
        try:
            creation_date = None
            mod_date = None
            
            # Extract dates from metadata
            if '/CreationDate' in metadata:
                creation_date = self._parse_pdf_date(metadata['/CreationDate'])
                
            if '/ModDate' in metadata:
                mod_date = self._parse_pdf_date(metadata['/ModDate'])
            
            if creation_date and mod_date:
                # Check if modification date is before creation date
                if mod_date < creation_date:
                    self.suspicious_indicators.append(
                        "Modification date is before creation date - clock manipulation suspected"
                    )
                    self.score += 0.6
                
                # Check for unusually large gap (>1 year)
                date_diff = (mod_date - creation_date).days
                if date_diff > 365:
                    self.suspicious_indicators.append(
                        f"Large time gap between creation and modification ({date_diff} days) - suspicious"
                    )
                    self.score += 0.3
                    
                # Check for modification in the future
                now = datetime.now()
                if mod_date > now:
                    self.suspicious_indicators.append(
                        "Modification date is in the future - clock manipulation suspected"
                    )
                    self.score += 0.7
                    
        except Exception as e:
            logger.warning(f"Could not check date consistency: {str(e)}")
    
    def _check_producer_consistency(self, metadata: Dict):
        """Check for inconsistent producer/creator strings"""
        try:
            producer = metadata.get('/Producer', '').lower()
            creator = metadata.get('/Creator', '').lower()
            
            # Common legitimate producers
            legitimate_producers = [
                'adobe', 'microsoft', 'libreoffice', 'openoffice', 
                'google', 'apple', 'foxit', 'nitro', 'pdftk'
            ]
            
            # Check for mixed or suspicious producers
            if producer and creator:
                # Look for obvious mismatches
                if 'adobe' in producer and 'microsoft' in creator:
                    self.suspicious_indicators.append(
                        "Producer/Creator mismatch suggests document assembly from multiple sources"
                    )
                    self.score += 0.3
            
            # Check for suspicious producer strings
            suspicious_patterns = ['unknown', 'modified', 'converted', 'temp', 'test']
            
            for pattern in suspicious_patterns:
                if pattern in producer or pattern in creator:
                    self.suspicious_indicators.append(
                        f"Suspicious producer/creator string contains '{pattern}'"
                    )
                    self.score += 0.2
                    
        except Exception as e:
            logger.warning(f"Could not check producer consistency: {str(e)}")
    
    def _check_document_id(self, metadata: Dict):
        """Check for missing or suspicious document ID"""
        try:
            if not metadata.get('has_document_id', False):
                self.suspicious_indicators.append(
                    "Document missing required /ID field - may indicate manual assembly"
                )
                self.score += 0.4
                
        except Exception as e:
            logger.warning(f"Could not check document ID: {str(e)}")
    
    def _check_object_structure(self, pdf: pikepdf.Pdf):
        """Check for suspicious object structure"""
        try:
            object_count = len(pdf.objects)
            
            # Very simple heuristic - real implementation would be more sophisticated
            if object_count < 5:
                self.suspicious_indicators.append(
                    f"Unusually few objects ({object_count}) for a business document"
                )
                self.score += 0.2
                
            # Check for suspicious object types (simplified)
            stream_objects = 0
            image_objects = 0
            
            for obj_id in pdf.objects:
                try:
                    obj = pdf.objects[obj_id]
                    if hasattr(obj, 'Type'):
                        if obj.Type == '/XObject' and hasattr(obj, 'Subtype'):
                            if obj.Subtype == '/Image':
                                image_objects += 1
                        elif hasattr(obj, 'stream'):
                            stream_objects += 1
                except:
                    continue
            
            # Check ratio of images to total objects
            if object_count > 0 and image_objects / object_count > 0.7:
                self.suspicious_indicators.append(
                    f"High ratio of image objects ({image_objects}/{object_count}) - may indicate scan-to-PDF conversion"
                )
                self.score += 0.2
                
        except Exception as e:
            logger.warning(f"Could not check object structure: {str(e)}")
    
    def _check_encryption_flags(self, pdf: pikepdf.Pdf, metadata: Dict):
        """Check for encryption-related flags"""
        try:
            if metadata.get('is_encrypted', False):
                # Document is encrypted - less suspicious but note it
                logger.info("Document is encrypted - limited metadata analysis possible")
            
            # Check for encryption removal traces (simplified)
            if '/Encrypt' in pdf.trailer:
                self.suspicious_indicators.append(
                    "Document shows signs of encryption handling - verify integrity"
                )
                self.score += 0.1
                
        except Exception as e:
            logger.warning(f"Could not check encryption flags: {str(e)}")
    
    def _has_digital_signature(self, pdf: pikepdf.Pdf) -> bool:
        """Check if document has digital signatures (simplified)"""
        try:
            # Look for signature fields
            if '/AcroForm' in pdf.Root:
                acroform = pdf.Root['/AcroForm']
                if '/Fields' in acroform:
                    fields = acroform['/Fields']
                    for field in fields:
                        try:
                            if '/FT' in field and field['/FT'] == '/Sig':
                                return True
                        except:
                            continue
            return False
        except:
            return False
    
    def _parse_pdf_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date string format (D:YYYYMMDDHHmmSSOHH'mm')"""
        try:
            # Remove D: prefix and timezone info for simple parsing
            date_clean = date_str.replace('D:', '')
            # Take first 14 characters (YYYYMMDDHHMMSS)
            if len(date_clean) >= 14:
                date_part = date_clean[:14]
                return datetime.strptime(date_part, '%Y%m%d%H%M%S')
            elif len(date_clean) >= 8:
                date_part = date_clean[:8]
                return datetime.strptime(date_part, '%Y%m%d')
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not parse date {date_str}: {str(e)}")
            return None


def test_pdf_analyzer():
    """Test function for the PDF metadata analyzer"""
    analyzer = PDFMetadataAnalyzer()
    logger.info("PDF metadata analyzer initialized successfully")


if __name__ == "__main__":
    test_pdf_analyzer() 