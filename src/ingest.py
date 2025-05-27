"""
Document Forgery Detection System - Ingest Module
Handles file loading and decides processing branch (born-digital vs scanned)
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pikepdf
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import logging
from loguru import logger

class DocumentIngestor:
    """Handles document ingestion and preprocessing"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Temporary directory created: {self.temp_dir}")
    
    def process_document(self, file_path: str) -> Dict:
        """
        Main processing function that routes documents through appropriate pipeline
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing processing results and metadata
        """
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                return self._process_pdf(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return self._process_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def _process_pdf(self, file_path: str) -> Dict:
        """Process PDF documents"""
        logger.info(f"Processing PDF: {file_path}")
        
        result = {
            'success': True,
            'file_path': file_path,
            'file_type': 'pdf',
            'processing_branch': None,
            'pages': [],
            'metadata': {},
            'has_text_layer': False,
            'page_count': 0
        }
        
        try:
            # Check if PDF has text layer using PyMuPDF
            doc = fitz.open(file_path)
            result['page_count'] = len(doc)
            
            # Check for text content
            text_content = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                text_content += page_text
            
            # Determine processing branch
            if len(text_content.strip()) > 50:  # Arbitrary threshold
                result['processing_branch'] = 'born_digital'
                result['has_text_layer'] = True
                logger.info("Document has text layer - using born-digital path")
            else:
                result['processing_branch'] = 'scanned_image'
                result['has_text_layer'] = False
                logger.info("Document appears to be scanned - using image path")
            
            # Convert pages to images for vision processing
            images = convert_from_path(file_path, dpi=300)
            
            for i, img in enumerate(images):
                # Save temporary image
                temp_img_path = os.path.join(self.temp_dir, f"page_{i+1}.png")
                img.save(temp_img_path, "PNG")
                
                result['pages'].append({
                    'page_number': i + 1,
                    'image_path': temp_img_path,
                    'image_size': img.size,
                    'text_content': doc[i].get_text() if result['has_text_layer'] else None
                })
            
            doc.close()
            
            # Extract PDF metadata using pikepdf
            try:
                with pikepdf.open(file_path) as pdf:
                    result['metadata'] = self._extract_pdf_metadata(pdf)
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {str(e)}")
                result['metadata'] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            result['success'] = False
            result['error'] = str(e)
            return result
    
    def _process_image(self, file_path: str) -> Dict:
        """Process standalone image files"""
        logger.info(f"Processing image: {file_path}")
        
        result = {
            'success': True,
            'file_path': file_path,
            'file_type': 'image',
            'processing_branch': 'scanned_image',
            'pages': [],
            'metadata': {},
            'has_text_layer': False,
            'page_count': 1
        }
        
        try:
            # Copy image to temp directory
            img = Image.open(file_path)
            temp_img_path = os.path.join(self.temp_dir, "page_1.png")
            img.save(temp_img_path, "PNG")
            
            result['pages'].append({
                'page_number': 1,
                'image_path': temp_img_path,
                'image_size': img.size,
                'text_content': None
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            result['success'] = False
            result['error'] = str(e)
            return result
    
    def _extract_pdf_metadata(self, pdf: pikepdf.Pdf) -> Dict:
        """Extract PDF metadata for forensic analysis"""
        metadata = {}
        
        try:
            # Basic document info
            if pdf.docinfo:
                for key, value in pdf.docinfo.items():
                    metadata[str(key)] = str(value)
            
            # Trailer information
            trailer = pdf.trailer
            if '/Info' in trailer:
                info = trailer['/Info']
                metadata['info_object'] = str(info)
            
            # Check for incremental saves
            metadata['has_prev_xref'] = '/Prev' in trailer
            if metadata['has_prev_xref']:
                metadata['prev_xref_offset'] = int(trailer['/Prev'])
                logger.info("Document has incremental saves - flagging for review")
            
            # Document ID
            if '/ID' in trailer:
                metadata['has_document_id'] = True
                metadata['document_id'] = str(trailer['/ID'])
            else:
                metadata['has_document_id'] = False
                logger.warning("Document missing /ID - potential security flag")
            
            # Version information
            metadata['pdf_version'] = str(pdf.pdf_version)
            
            # Object count
            metadata['object_count'] = len(pdf.objects)
            
            # Check for encryption
            metadata['is_encrypted'] = pdf.is_encrypted
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {}
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temp directory: {str(e)}")


def test_ingestor():
    """Test function for the document ingestor"""
    ingestor = DocumentIngestor()
    
    # This would be used with actual test files
    logger.info("Document ingestor initialized successfully")
    
    ingestor.cleanup()


if __name__ == "__main__":
    test_ingestor() 