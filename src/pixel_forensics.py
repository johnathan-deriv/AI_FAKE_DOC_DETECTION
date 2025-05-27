"""
Document Forgery Detection System - Enhanced Pixel-Level Forensics Module
Module 2C: Advanced pixel-level analysis for tampering detection

This module implements sophisticated pixel-level analysis techniques including:
- Block-based statistical analysis
- Advanced JPEG compression artifact detection with DCT coefficient analysis
- Copy-move forgery detection using SIFT/ORB features
- Wavelet decomposition analysis
- Noise pattern analysis with multi-scale approach
- Edge and gradient inconsistencies
- Color space anomalies
- Frequency domain tampering detection
- Multi-resolution analysis

Optimized for both fast web applications and thorough forensic analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import scipy.fft
from scipy import stats, ndimage
from scipy.ndimage import gaussian_filter
from skimage import feature, measure, filters
from loguru import logger
import os
from dataclasses import dataclass
import json
import pywt  # For wavelet analysis


@dataclass
class PixelBlock:
    """Represents a pixel block with its properties"""
    x: int
    y: int
    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    edge_density: float
    texture_energy: float
    compression_score: float
    noise_level: float
    
    
@dataclass
class AnomalyRegion:
    """Represents a detected anomalous region"""
    x: int
    y: int
    width: int
    height: int
    anomaly_type: str
    confidence: float
    description: str


@dataclass
class CopyMoveRegion:
    """Represents a detected copy-move forgery region"""
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    width: int
    height: int
    confidence: float
    similarity_score: float
    feature_count: int
    
    
@dataclass 
class FrequencyAnalysis:
    """Results from frequency domain analysis"""
    dct_irregularities: List[Tuple[int, int, float]]  # (x, y, anomaly_score)
    wavelet_anomalies: List[Tuple[int, int, str, float]]  # (x, y, wavelet_type, score)
    power_spectrum_anomalies: List[float]
    frequency_tampering_score: float


class PixelForensicsAnalyzer:
    """Advanced pixel-level analysis for document tampering detection - Enhanced Version"""
    
    def __init__(self, block_size: int = 16, fast_mode: bool = True, enhanced_mode: Union[bool, str] = False):
        """
        Initialize pixel forensics analyzer
        
        Args:
            block_size: Size of pixel blocks for analysis (default 16x16 for speed)
            fast_mode: Enable optimizations for web application use
            enhanced_mode: Enhanced forensics mode - can be:
                          - False/disabled: No enhanced features
                          - True/basic_fast: Standard enhanced features 
                          - "enhanced_fast": DCT analysis, basic wavelets, copy-move detection (optimized)
                          - "enhanced_thorough": Full analysis with all techniques
        """
        self.block_size = block_size
        self.fast_mode = fast_mode
        
        # Handle different enhanced_mode values
        if enhanced_mode == "disabled":
            # Disabled mode - minimal pixel analysis
            self.enhanced_mode = False
            self.enhanced_mode_type = "disabled"
        elif enhanced_mode == "basic_fast" or enhanced_mode is False:
            # Basic fast mode - standard pixel analysis without advanced features
            self.enhanced_mode = False
            self.enhanced_mode_type = "basic_fast"
        elif enhanced_mode is True:
            self.enhanced_mode = True
            self.enhanced_mode_type = "enhanced_fast"  # Default enhanced mode
        elif enhanced_mode in ["enhanced_fast", "enhanced_thorough"]:
            self.enhanced_mode = True
            self.enhanced_mode_type = enhanced_mode
        else:
            # Default fallback
            self.enhanced_mode = False
            self.enhanced_mode_type = "basic_fast"
        
        self.anomaly_threshold = 2.5  # Standard deviations for anomaly detection
        
        # Configure analysis parameters based on mode
        if self.enhanced_mode_type == "disabled":
            # Disabled mode - minimal analysis
            self.max_image_size = 512
            self.sample_ratio = 0.1    # Very limited sampling
            self.skip_expensive_ops = True
        elif self.enhanced_mode_type == "basic_fast":
            # Basic fast mode - reasonable analysis with optimizations
            self.max_image_size = 1024  # Resize large images for speed
            self.sample_ratio = 0.3     # Sample 30% of blocks for analysis
            self.skip_expensive_ops = True  # Skip GLCM and complex texture analysis
        elif self.enhanced_mode_type == "enhanced_fast":
            # Enhanced fast mode - optimized for web apps with reduced anomaly density
            self.max_image_size = 1024
            self.sample_ratio = 0.5     # More samples for enhanced accuracy
            self.skip_expensive_ops = False
            self.anomaly_threshold = 3.0  # Higher threshold to reduce false positives
        elif self.enhanced_mode_type == "enhanced_thorough":
            # Enhanced thorough mode - maximum accuracy
            self.max_image_size = 2048 if fast_mode else None
            self.sample_ratio = 0.8 if fast_mode else 1.0
            self.skip_expensive_ops = False
            self.anomaly_threshold = 2.5  # Standard threshold for thorough analysis
        else:
            # Default values
            self.max_image_size = None
            self.sample_ratio = 1.0
            self.skip_expensive_ops = False
            
        # Enhanced mode features
        if self.enhanced_mode:
            logger.info(f"Enhanced pixel forensics mode '{self.enhanced_mode_type}' enabled with advanced techniques")
            # Initialize SIFT detector for copy-move detection
            try:
                self.sift_detector = cv2.SIFT_create(nfeatures=500)
                self.orb_detector = cv2.ORB_create(nfeatures=500)
                logger.info("Feature detectors initialized for copy-move analysis")
            except Exception as e:
                logger.warning(f"Could not initialize feature detectors: {e}")
                self.sift_detector = None
                self.orb_detector = None
        else:
            logger.info(f"Pixel forensics mode: {self.enhanced_mode_type}")
            self.sift_detector = None
            self.orb_detector = None
        
    def analyze_image(self, image_path: str) -> Dict:
        """
        Perform comprehensive pixel-level analysis with enhanced techniques
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            original_height, original_width = image.shape[:2]
            scale_factor = 1.0  # Track scaling for coordinate conversion
            
            # Resize if needed
            if not self.fast_mode or max(original_height, original_width) <= self.max_image_size:
                # Use original image
                pass
            else:
                if max(original_height, original_width) > self.max_image_size:
                    scale_factor = self.max_image_size / max(original_height, original_width)
                    new_width = int(original_width * scale_factor)
                    new_height = int(original_height * scale_factor)
                    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height} for faster analysis")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            logger.info(f"Starting {'enhanced' if self.enhanced_mode else 'optimized'} pixel-level analysis of {image_path} with block size {self.block_size}x{self.block_size}")
            
            # Perform core analyses
            block_analysis = self._analyze_blocks_fast(gray)
            compression_analysis = self._analyze_compression_artifacts_enhanced(gray) if self.enhanced_mode else self._analyze_compression_artifacts_fast(gray)
            noise_analysis = self._analyze_noise_patterns_enhanced(gray) if self.enhanced_mode else self._analyze_noise_patterns_fast(gray)
            edge_analysis = self._analyze_edge_consistency_fast(gray)
            color_analysis = self._analyze_color_inconsistencies_fast(image)
            
            # Enhanced mode additional analyses
            copy_move_regions = []
            frequency_analysis = None
            wavelet_analysis = {}
            
            if self.enhanced_mode:
                logger.info("Running enhanced forensic analyses...")
                
                # Copy-move forgery detection
                copy_move_regions = self._detect_copy_move_forgery(gray)
                
                # Advanced frequency domain analysis
                frequency_analysis = self._analyze_frequency_domain(gray)
                
                # Wavelet decomposition analysis
                wavelet_analysis = self._analyze_wavelet_decomposition(gray)
            
            # Detect anomalous regions (enhanced or basic)
            if self.enhanced_mode:
                anomaly_regions = self._detect_anomalous_regions_enhanced(
                    block_analysis, gray.shape, frequency_analysis, copy_move_regions
                )
            else:
                anomaly_regions = self._detect_anomalous_regions_fast(
                    block_analysis, gray.shape
                )
            
            # Scale coordinates back to original image size if image was resized
            if scale_factor != 1.0:
                for region in anomaly_regions:
                    region.x = int(region.x / scale_factor)
                    region.y = int(region.y / scale_factor)
                    region.width = int(region.width / scale_factor)
                    region.height = int(region.height / scale_factor)
                
                # Scale copy-move regions
                for cm_region in copy_move_regions:
                    cm_region.source_x = int(cm_region.source_x / scale_factor)
                    cm_region.source_y = int(cm_region.source_y / scale_factor)
                    cm_region.target_x = int(cm_region.target_x / scale_factor)
                    cm_region.target_y = int(cm_region.target_y / scale_factor)
                    cm_region.width = int(cm_region.width / scale_factor)
                    cm_region.height = int(cm_region.height / scale_factor)
            
            # Calculate overall suspicion score
            if self.enhanced_mode:
                score = self._calculate_enhanced_pixel_score(
                    block_analysis, compression_analysis, noise_analysis, 
                    edge_analysis, color_analysis, anomaly_regions,
                    copy_move_regions, frequency_analysis, wavelet_analysis
                )
            else:
                score = self._calculate_pixel_score(
                    block_analysis, compression_analysis, noise_analysis, 
                    edge_analysis, color_analysis, anomaly_regions
                )
            
            result = {
                'module': 'pixel_forensics_enhanced' if self.enhanced_mode else 'pixel_forensics',
                'score': score,
                'suspicious': score > 0.4,
                'block_size': self.block_size,
                'fast_mode': self.fast_mode,
                'enhanced_mode': self.enhanced_mode,
                'anomaly_regions': [
                    {
                        'x': region.x, 'y': region.y, 
                        'width': region.width, 'height': region.height,
                        'type': region.anomaly_type,
                        'confidence': region.confidence,
                        'description': region.description
                    }
                    for region in anomaly_regions
                ],
                'indicators': self._generate_enhanced_indicators(
                    compression_analysis, noise_analysis, edge_analysis, 
                    color_analysis, anomaly_regions, copy_move_regions,
                    frequency_analysis, wavelet_analysis
                ) if self.enhanced_mode else self._generate_indicators(
                    compression_analysis, noise_analysis, edge_analysis, 
                    color_analysis, anomaly_regions
                ),
                'details': {
                    'image_analyzed': image_path,
                    'total_blocks': len(block_analysis),
                    'anomalous_blocks': len([b for b in block_analysis if self._is_block_anomalous(b, block_analysis)]),
                    'compression_artifacts': compression_analysis['artifact_count'],
                    'noise_inconsistencies': noise_analysis['inconsistent_regions'],
                    'edge_discontinuities': edge_analysis['discontinuity_count'],
                    'color_anomalies': color_analysis['anomaly_count']
                }
            }
            
            # Add enhanced mode results
            if self.enhanced_mode:
                result['copy_move_regions'] = [
                    {
                        'source_x': cm.source_x, 'source_y': cm.source_y,
                        'target_x': cm.target_x, 'target_y': cm.target_y,
                        'width': cm.width, 'height': cm.height,
                        'confidence': cm.confidence,
                        'similarity_score': cm.similarity_score,
                        'feature_count': cm.feature_count
                    }
                    for cm in copy_move_regions
                ]
                
                if frequency_analysis:
                    result['frequency_analysis'] = {
                        'dct_irregularities_count': len(frequency_analysis.dct_irregularities),
                        'wavelet_anomalies_count': len(frequency_analysis.wavelet_anomalies),
                        'frequency_tampering_score': frequency_analysis.frequency_tampering_score
                    }
                
                result['wavelet_analysis'] = wavelet_analysis
                result['details'].update({
                    'copy_move_detections': len(copy_move_regions),
                    'dct_irregularities': len(frequency_analysis.dct_irregularities) if frequency_analysis else 0,
                    'wavelet_anomalies': len(frequency_analysis.wavelet_anomalies) if frequency_analysis else 0
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in pixel-level analysis: {str(e)}")
            return {
                'module': 'pixel_forensics_enhanced' if self.enhanced_mode else 'pixel_forensics',
                'score': 0.3,  # Default suspicious score for errors
                'suspicious': True,
                'error': str(e),
                'indicators': [f"Pixel analysis failed: {str(e)}"],
                'details': {}
            }
    
    def _analyze_blocks_fast(self, gray_image: np.ndarray) -> List[PixelBlock]:
        """Optimized block analysis using vectorized operations"""
        blocks = []
        height, width = gray_image.shape
        
        # Pre-compute edge image once
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Pre-compute noise image once
        noise_image = self._compute_noise_image_fast(gray_image)
        
        block_count = 0
        total_possible_blocks = ((height - self.block_size + 1) // self.block_size) * ((width - self.block_size + 1) // self.block_size)
        
        for y in range(0, height - self.block_size + 1, self.block_size):
            for x in range(0, width - self.block_size + 1, self.block_size):
                # Sample blocks for detailed analysis in fast mode
                if self.fast_mode and np.random.random() > self.sample_ratio:
                    continue
                
                # Extract block regions efficiently
                block = gray_image[y:y+self.block_size, x:x+self.block_size]
                edge_block = edge_magnitude[y:y+self.block_size, x:x+self.block_size]
                noise_block = noise_image[y:y+self.block_size, x:x+self.block_size]
                
                # Vectorized calculations
                mean_intensity = np.mean(block)
                std_intensity = np.std(block)
                edge_density = np.mean(edge_block)
                noise_level = np.mean(noise_block)
                
                # Fast texture energy (simplified)
                if self.skip_expensive_ops:
                    texture_energy = np.std(block) / (np.mean(block) + 1e-10)  # Simple texture measure
                else:
                    texture_energy = self._calculate_texture_energy(block)
                
                # Fast compression score
                compression_score = self._calculate_block_compression_score_fast(block)
                
                blocks.append(PixelBlock(
                    x=x, y=y, width=self.block_size, height=self.block_size,
                    mean_intensity=mean_intensity, std_intensity=std_intensity,
                    edge_density=edge_density, texture_energy=texture_energy,
                    compression_score=compression_score, noise_level=noise_level
                ))
                block_count += 1
        
        logger.info(f"Analyzed {len(blocks)} pixel blocks (sampled from {total_possible_blocks} total blocks)")
        return blocks
    
    def _compute_noise_image_fast(self, gray_image: np.ndarray) -> np.ndarray:
        """Fast noise computation using simple Laplacian"""
        # Use simple Laplacian for speed
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
        return np.abs(laplacian) / 255.0  # Normalize
    
    def _calculate_texture_energy(self, block: np.ndarray) -> float:
        """Calculate texture energy using GLCM (only in non-fast mode)"""
        if self.skip_expensive_ops:
            # Simple texture measure
            return np.std(block) / (np.mean(block) + 1e-10)
        
        try:
            # Normalize block to 0-255 range
            block_normalized = ((block - block.min()) * 255 / (block.max() - block.min() + 1e-10)).astype(np.uint8)
            
            # Calculate GLCM (simplified)
            glcm = feature.graycomatrix(block_normalized, [1], [0], levels=256, symmetric=True, normed=True)
            
            # Calculate energy (Angular Second Moment)
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
            return float(energy)
        except:
            return 0.0
    
    def _calculate_block_compression_score_fast(self, block: np.ndarray) -> float:
        """Fast compression artifacts detection"""
        try:
            # Simple frequency domain analysis
            block_float = block.astype(np.float32)
            
            # Use simple FFT instead of DCT for speed
            fft_block = np.fft.fft2(block_float)
            fft_magnitude = np.abs(fft_block)
            
            # Check high vs low frequency energy ratio
            h, w = fft_magnitude.shape
            mid_h, mid_w = h // 2, w // 2
            
            low_freq = np.mean(fft_magnitude[:mid_h//2, :mid_w//2])
            high_freq = np.mean(fft_magnitude[mid_h//2:, mid_w//2:])
            
            if low_freq > 0:
                ratio = high_freq / low_freq
                return 1.0 - min(ratio, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _analyze_compression_artifacts_fast(self, gray_image: np.ndarray) -> Dict:
        """Fast compression artifacts detection"""
        try:
            # Apply DCT to entire image in 8x8 blocks
            height, width = gray_image.shape
            artifact_map = np.zeros((height // 8, width // 8))
            artifact_count = 0
            
            for i in range(0, height - 7, 8):
                for j in range(0, width - 7, 8):
                    block = gray_image[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Check for blocking artifacts at block boundaries
                    if i > 0:  # Check vertical boundary
                        boundary_diff = np.abs(gray_image[i-1, j:j+8].astype(float) - gray_image[i, j:j+8].astype(float))
                        if np.mean(boundary_diff) > 10:  # Threshold for boundary artifact
                            artifact_count += 1
                            artifact_map[i//8, j//8] = 1
                    
                    if j > 0:  # Check horizontal boundary
                        boundary_diff = np.abs(gray_image[i:i+8, j-1].astype(float) - gray_image[i:i+8, j].astype(float))
                        if np.mean(boundary_diff) > 10:
                            artifact_count += 1
                            artifact_map[i//8, j//8] = 1
            
            return {
                'artifact_count': artifact_count,
                'artifact_density': artifact_count / (artifact_map.size + 1e-10),
                'artifact_map': artifact_map
            }
            
        except Exception as e:
            logger.warning(f"Error in compression analysis: {str(e)}")
            return {'artifact_count': 0, 'artifact_density': 0.0, 'artifact_map': None}
    
    def _analyze_noise_patterns_fast(self, gray_image: np.ndarray) -> Dict:
        """Fast noise analysis using simple Laplacian"""
        try:
            # Apply Gaussian filter to estimate noise
            filtered = gaussian_filter(gray_image, sigma=1.0)
            noise = gray_image.astype(float) - filtered
            
            # Divide image into regions and analyze noise statistics
            h, w = noise.shape
            region_size = max(h // 8, w // 8, 32)  # Adaptive region size
            noise_stats = []
            inconsistent_regions = 0
            
            for i in range(0, h - region_size + 1, region_size):
                for j in range(0, w - region_size + 1, region_size):
                    region_noise = noise[i:i+region_size, j:j+region_size]
                    noise_std = np.std(region_noise)
                    noise_stats.append(noise_std)
            
            if len(noise_stats) > 1:
                noise_mean = np.mean(noise_stats)
                noise_threshold = noise_mean + 2 * np.std(noise_stats)
                
                # Count regions with significantly different noise
                inconsistent_regions = sum(1 for std in noise_stats if std > noise_threshold)
            
            return {
                'noise_mean': float(np.mean(noise_stats)) if noise_stats else 0.0,
                'noise_std': float(np.std(noise_stats)) if noise_stats else 0.0,
                'inconsistent_regions': inconsistent_regions,
                'total_regions': len(noise_stats)
            }
            
        except Exception as e:
            logger.warning(f"Error in noise analysis: {str(e)}")
            return {'noise_mean': 0.0, 'noise_std': 0.0, 'inconsistent_regions': 0, 'total_regions': 0}
    
    def _analyze_edge_consistency_fast(self, gray_image: np.ndarray) -> Dict:
        """Fast edge analysis using Canny"""
        try:
            # Detect edges using Canny
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find edge contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            discontinuity_count = 0
            total_contours = len(contours)
            
            # Analyze each contour for discontinuities
            for contour in contours:
                if len(contour) > 10:  # Only analyze substantial contours
                    # Calculate curvature changes
                    contour_points = contour.reshape(-1, 2)
                    
                    # Check for abrupt direction changes (potential tampering indicators)
                    for i in range(2, len(contour_points) - 2):
                        # Calculate vectors
                        v1 = contour_points[i] - contour_points[i-1]
                        v2 = contour_points[i+1] - contour_points[i]
                        
                        # Calculate angle between vectors
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        
                        # Detect sharp turns (potential discontinuities)
                        if angle > np.pi * 0.7:  # > 126 degrees
                            discontinuity_count += 1
            
            return {
                'discontinuity_count': discontinuity_count,
                'total_contours': total_contours,
                'edge_density': np.sum(edges > 0) / edges.size
            }
            
        except Exception as e:
            logger.warning(f"Error in edge analysis: {str(e)}")
            return {'discontinuity_count': 0, 'total_contours': 0, 'edge_density': 0.0}
    
    def _analyze_color_inconsistencies_fast(self, color_image: np.ndarray) -> Dict:
        """Fast color analysis using simple statistics"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
            
            anomaly_count = 0
            
            # Analyze color distribution in regions
            h, w = color_image.shape[:2]
            region_size = max(h // 6, w // 6, 64)
            
            hue_stats = []
            saturation_stats = []
            lightness_stats = []
            
            for i in range(0, h - region_size + 1, region_size):
                for j in range(0, w - region_size + 1, region_size):
                    hsv_region = hsv[i:i+region_size, j:j+region_size]
                    lab_region = lab[i:i+region_size, j:j+region_size]
                    
                    # Calculate statistics for each channel
                    hue_std = np.std(hsv_region[:, :, 0])
                    sat_std = np.std(hsv_region[:, :, 1])
                    light_std = np.std(lab_region[:, :, 0])
                    
                    hue_stats.append(hue_std)
                    saturation_stats.append(sat_std)
                    lightness_stats.append(light_std)
            
            # Detect anomalous regions
            if len(hue_stats) > 1:
                hue_threshold = np.mean(hue_stats) + 2 * np.std(hue_stats)
                sat_threshold = np.mean(saturation_stats) + 2 * np.std(saturation_stats)
                light_threshold = np.mean(lightness_stats) + 2 * np.std(lightness_stats)
                
                for i, (h_std, s_std, l_std) in enumerate(zip(hue_stats, saturation_stats, lightness_stats)):
                    if h_std > hue_threshold or s_std > sat_threshold or l_std > light_threshold:
                        anomaly_count += 1
            
            return {
                'anomaly_count': anomaly_count,
                'total_regions': len(hue_stats),
                'hue_inconsistency': float(np.std(hue_stats)) if hue_stats else 0.0,
                'saturation_inconsistency': float(np.std(saturation_stats)) if saturation_stats else 0.0,
                'lightness_inconsistency': float(np.std(lightness_stats)) if lightness_stats else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Error in color analysis: {str(e)}")
            return {'anomaly_count': 0, 'total_regions': 0, 'hue_inconsistency': 0.0, 
                   'saturation_inconsistency': 0.0, 'lightness_inconsistency': 0.0}
    
    def _detect_anomalous_regions_fast(self, blocks: List[PixelBlock], image_shape: Tuple[int, int]) -> List[AnomalyRegion]:
        """Detect anomalous regions based on block analysis"""
        anomalous_regions = []
        
        if not blocks:
            return anomalous_regions
        
        # Calculate statistics for different block properties
        intensities = [b.mean_intensity for b in blocks]
        stds = [b.std_intensity for b in blocks]
        edge_densities = [b.edge_density for b in blocks]
        texture_energies = [b.texture_energy for b in blocks]
        compression_scores = [b.compression_score for b in blocks]
        noise_levels = [b.noise_level for b in blocks]
        
        # Calculate thresholds for anomaly detection
        intensity_threshold = np.mean(intensities) + self.anomaly_threshold * np.std(intensities)
        std_threshold = np.mean(stds) + self.anomaly_threshold * np.std(stds)
        edge_threshold = np.mean(edge_densities) + self.anomaly_threshold * np.std(edge_densities)
        texture_threshold = np.mean(texture_energies) + self.anomaly_threshold * np.std(texture_energies)
        compression_threshold = np.mean(compression_scores) + self.anomaly_threshold * np.std(compression_scores)
        noise_threshold = np.mean(noise_levels) + self.anomaly_threshold * np.std(noise_levels)
        
        # Identify anomalous blocks
        for block in blocks:
            anomalies = []
            confidence = 0.0
            
            if block.mean_intensity > intensity_threshold:
                anomalies.append("unusual_brightness")
                confidence += 0.2
                
            if block.std_intensity > std_threshold:
                anomalies.append("high_variance")
                confidence += 0.15
                
            if block.edge_density > edge_threshold:
                anomalies.append("excessive_edges")
                confidence += 0.2
                
            if block.texture_energy > texture_threshold:
                anomalies.append("texture_anomaly")
                confidence += 0.15
                
            if block.compression_score > compression_threshold:
                anomalies.append("compression_inconsistency")
                confidence += 0.2
                
            if block.noise_level > noise_threshold:
                anomalies.append("noise_anomaly")
                confidence += 0.1
            
            # Adjust minimum confidence threshold based on mode
            min_confidence = 0.4 if self.enhanced_mode_type == "basic_fast" else 0.3
            
            if anomalies and confidence > min_confidence:
                anomalous_regions.append(AnomalyRegion(
                    x=block.x, y=block.y, width=block.width, height=block.height,
                    anomaly_type=", ".join(anomalies),
                    confidence=min(confidence, 1.0),
                    description=f"Block shows {', '.join(anomalies)} (confidence: {confidence:.2f})"
                ))
        
        logger.info(f"Detected {len(anomalous_regions)} anomalous regions using {'basic fast' if self.enhanced_mode_type == 'basic_fast' else 'fast'} analysis")
        return anomalous_regions
    
    def _detect_anomalous_regions_enhanced(self, blocks: List[PixelBlock], image_shape: Tuple[int, int], 
                                         frequency_analysis: Optional[FrequencyAnalysis],
                                         copy_move_regions: List[CopyMoveRegion]) -> List[AnomalyRegion]:
        """Enhanced anomaly detection incorporating frequency domain and copy-move analysis"""
        # Start with basic block-based anomalies
        anomalous_regions = self._detect_anomalous_regions_fast(blocks, image_shape)
        
        # Add frequency domain anomalies with higher confidence thresholds
        if frequency_analysis:
            # Increase confidence threshold for DCT irregularities
            dct_threshold = 0.7 if self.enhanced_mode_type == "enhanced_fast" else 0.6
            for x, y, score in frequency_analysis.dct_irregularities:
                if score > dct_threshold:
                    anomalous_regions.append(AnomalyRegion(
                        x=x, y=y, width=8, height=8,  # DCT block size
                        anomaly_type="dct_irregularity",
                        confidence=min(score, 1.0),
                        description=f"DCT coefficient irregularity detected (score: {score:.2f})"
                    ))
            
            # Increase confidence threshold for wavelet anomalies
            wavelet_threshold = 0.7 if self.enhanced_mode_type == "enhanced_fast" else 0.6
            for x, y, wavelet_type, score in frequency_analysis.wavelet_anomalies:
                if score > wavelet_threshold:
                    anomalous_regions.append(AnomalyRegion(
                        x=x, y=y, width=16, height=16,  # Wavelet region size
                        anomaly_type=f"wavelet_anomaly_{wavelet_type}",
                        confidence=min(score, 1.0),
                        description=f"Wavelet anomaly in {wavelet_type} (score: {score:.2f})"
                    ))
        
        # Add copy-move regions as anomalies with higher confidence threshold
        cm_threshold = 0.6 if self.enhanced_mode_type == "enhanced_fast" else 0.5
        for cm_region in copy_move_regions:
            if cm_region.confidence > cm_threshold:
                # Add both source and target regions
                anomalous_regions.append(AnomalyRegion(
                    x=cm_region.source_x, y=cm_region.source_y,
                    width=cm_region.width, height=cm_region.height,
                    anomaly_type="copy_move_source",
                    confidence=cm_region.confidence,
                    description=f"Copy-move source region ({cm_region.feature_count} features)"
                ))
                
                anomalous_regions.append(AnomalyRegion(
                    x=cm_region.target_x, y=cm_region.target_y,
                    width=cm_region.width, height=cm_region.height,
                    anomaly_type="copy_move_target",
                    confidence=cm_region.confidence,
                    description=f"Copy-move target region ({cm_region.feature_count} features)"
                ))
        
        # Remove overlapping regions and filter by confidence
        filtered_regions = self._filter_overlapping_regions(anomalous_regions)
        
        # Additional filtering for enhanced fast mode to reduce visual clutter
        if self.enhanced_mode_type == "enhanced_fast":
            # Keep only top confidence regions and limit total number
            high_confidence_regions = [r for r in filtered_regions if r.confidence > 0.6]
            if len(high_confidence_regions) > 20:  # Limit to top 20 regions
                high_confidence_regions.sort(key=lambda r: r.confidence, reverse=True)
                filtered_regions = high_confidence_regions[:20]
            else:
                filtered_regions = high_confidence_regions
        
        logger.info(f"Enhanced detection found {len(filtered_regions)} anomalous regions (filtered from {len(anomalous_regions)})")
        return filtered_regions
    
    def _filter_overlapping_regions(self, regions: List[AnomalyRegion]) -> List[AnomalyRegion]:
        """Filter overlapping anomalous regions, keeping highest confidence ones"""
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence (highest first)
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        filtered_regions = []
        
        for region in sorted_regions:
            overlaps = False
            
            for existing_region in filtered_regions:
                # Check for overlap
                if self._regions_overlap(region, existing_region):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _regions_overlap(self, region1: AnomalyRegion, region2: AnomalyRegion, 
                        threshold: float = 0.3) -> bool:
        """Check if two regions overlap by more than threshold"""
        try:
            # Calculate intersection
            x1 = max(region1.x, region2.x)
            y1 = max(region1.y, region2.y)
            x2 = min(region1.x + region1.width, region2.x + region2.width)
            y2 = min(region1.y + region1.height, region2.y + region2.height)
            
            if x2 <= x1 or y2 <= y1:
                return False  # No overlap
            
            # Calculate overlap area
            overlap_area = (x2 - x1) * (y2 - y1)
            
            # Calculate areas of both regions
            area1 = region1.width * region1.height
            area2 = region2.width * region2.height
            
            # Calculate overlap ratio
            overlap_ratio = overlap_area / min(area1, area2)
            
            return overlap_ratio > threshold
            
        except Exception:
            return False

    def _is_block_anomalous(self, block: PixelBlock, all_blocks: List[PixelBlock]) -> bool:
        """Check if a block is anomalous compared to others"""
        if not all_blocks:
            return False
        
        # Simple statistical check
        intensities = [b.mean_intensity for b in all_blocks]
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        # Block is anomalous if it's more than 2 standard deviations away
        return abs(block.mean_intensity - mean_intensity) > 2 * std_intensity
    
    def _calculate_enhanced_pixel_score(self, blocks: List[PixelBlock], compression_analysis: Dict, 
                                      noise_analysis: Dict, edge_analysis: Dict, 
                                      color_analysis: Dict, anomaly_regions: List[AnomalyRegion],
                                      copy_move_regions: List[CopyMoveRegion],
                                      frequency_analysis: Optional[FrequencyAnalysis],
                                      wavelet_analysis: Dict) -> float:
        """Calculate enhanced overall pixel-level suspicion score"""
        score = 0.0
        
        # Base score from anomalous blocks (reduced weight in enhanced mode)
        if blocks:
            anomalous_count = len([b for b in blocks if self._is_block_anomalous(b, blocks)])
            anomaly_ratio = anomalous_count / len(blocks)
            score += anomaly_ratio * 0.25  # Reduced from 0.4
        
        # Compression artifacts
        compression_density = compression_analysis.get('artifact_density', 0.0)
        if compression_density > 0.1:  # 10% threshold
            enhanced_weight = 0.35 if compression_analysis.get('enhanced_analysis') else 0.3
            score += min(compression_density * enhanced_weight, enhanced_weight)
        
        # Noise inconsistencies
        noise_ratio = 0.0
        if noise_analysis.get('total_regions', 0) > 0:
            noise_ratio = noise_analysis.get('inconsistent_regions', 0) / noise_analysis['total_regions']
        if noise_ratio > 0.2:  # 20% threshold
            enhanced_weight = 0.25 if noise_analysis.get('enhanced_analysis') else 0.2
            score += min(noise_ratio * enhanced_weight, enhanced_weight)
        
        # Enhanced noise inconsistencies from wavelet analysis
        if noise_analysis.get('wavelet_noise_inconsistencies', 0) > 0:
            wavelet_noise_score = noise_analysis.get('wavelet_noise_inconsistencies', 0) / 3.0  # Normalize
            score += min(wavelet_noise_score * 0.15, 0.15)
        
        # Edge discontinuities
        edge_density = edge_analysis.get('edge_density', 0.0)
        discontinuity_count = edge_analysis.get('discontinuity_count', 0)
        if discontinuity_count > 10 and edge_density > 0.1:
            score += min(discontinuity_count / 100.0 * 0.15, 0.15)  # Reduced weight
        
        # Color inconsistencies
        color_ratio = 0.0
        if color_analysis.get('total_regions', 0) > 0:
            color_ratio = color_analysis.get('anomaly_count', 0) / color_analysis['total_regions']
        if color_ratio > 0.15:  # 15% threshold
            score += min(color_ratio * 0.1, 0.1)  # Reduced weight
        
        # Enhanced forensic techniques scores
        if self.enhanced_mode:
            # Copy-move forgery
            if copy_move_regions:
                cm_score = sum(region.confidence for region in copy_move_regions) / len(copy_move_regions)
                cm_weight = min(len(copy_move_regions) / 3.0, 1.0)  # More regions = higher weight
                score += cm_score * cm_weight * 0.3
            
            # Frequency domain analysis
            if frequency_analysis:
                freq_score = frequency_analysis.frequency_tampering_score
                score += freq_score * 0.25
            
            # Wavelet analysis
            if wavelet_analysis.get('anomaly_regions'):
                wavelet_score = len(wavelet_analysis['anomaly_regions']) / 5.0  # Normalize
                score += min(wavelet_score * 0.2, 0.2)
        
        # Anomalous regions contribution (enhanced)
        if anomaly_regions:
            # Weight by type of anomaly
            weighted_score = 0.0
            for region in anomaly_regions:
                weight = 1.0
                if 'copy_move' in region.anomaly_type:
                    weight = 1.5  # Copy-move is strong evidence
                elif 'dct' in region.anomaly_type or 'wavelet' in region.anomaly_type:
                    weight = 1.3  # Frequency domain anomalies are significant
                
                weighted_score += region.confidence * weight
            
            avg_weighted_score = weighted_score / len(anomaly_regions)
            score += min(avg_weighted_score * 0.2, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_pixel_score(self, blocks: List[PixelBlock], compression_analysis: Dict, 
                             noise_analysis: Dict, edge_analysis: Dict, 
                             color_analysis: Dict, anomaly_regions: List[AnomalyRegion]) -> float:
        """Calculate overall pixel-level suspicion score (basic version)"""
        score = 0.0
        
        # Base score from anomalous blocks
        if blocks:
            anomalous_count = len([b for b in blocks if self._is_block_anomalous(b, blocks)])
            anomaly_ratio = anomalous_count / len(blocks)
            score += anomaly_ratio * 0.4
        
        # Compression artifacts
        compression_density = compression_analysis.get('artifact_density', 0.0)
        if compression_density > 0.1:  # 10% threshold
            score += min(compression_density * 0.3, 0.3)
        
        # Noise inconsistencies
        noise_ratio = 0.0
        if noise_analysis.get('total_regions', 0) > 0:
            noise_ratio = noise_analysis.get('inconsistent_regions', 0) / noise_analysis['total_regions']
        if noise_ratio > 0.2:  # 20% threshold
            score += min(noise_ratio * 0.2, 0.2)
        
        # Edge discontinuities
        edge_density = edge_analysis.get('edge_density', 0.0)
        discontinuity_count = edge_analysis.get('discontinuity_count', 0)
        if discontinuity_count > 10 and edge_density > 0.1:
            score += min(discontinuity_count / 100.0 * 0.2, 0.2)
        
        # Color inconsistencies
        color_ratio = 0.0
        if color_analysis.get('total_regions', 0) > 0:
            color_ratio = color_analysis.get('anomaly_count', 0) / color_analysis['total_regions']
        if color_ratio > 0.15:  # 15% threshold
            score += min(color_ratio * 0.15, 0.15)
        
        # Anomalous regions contribution
        if anomaly_regions:
            region_score = sum(region.confidence for region in anomaly_regions) / len(anomaly_regions)
            score += min(region_score * 0.25, 0.25)
        
        return min(score, 1.0)
    
    def _generate_enhanced_indicators(self, compression_analysis: Dict, noise_analysis: Dict,
                                    edge_analysis: Dict, color_analysis: Dict,
                                    anomaly_regions: List[AnomalyRegion],
                                    copy_move_regions: List[CopyMoveRegion],
                                    frequency_analysis: Optional[FrequencyAnalysis],
                                    wavelet_analysis: Dict) -> List[str]:
        """Generate enhanced human-readable indicators"""
        indicators = []
        
        # Basic indicators (enhanced versions where applicable)
        if compression_analysis.get('artifact_density', 0) > 0.1:
            enhanced_note = " (enhanced DCT analysis)" if compression_analysis.get('enhanced_analysis') else ""
            indicators.append(f"JPEG compression artifacts detected in {compression_analysis['artifact_count']} regions{enhanced_note}")
        
        # Enhanced noise analysis
        noise_ratio = 0.0
        if noise_analysis.get('total_regions', 0) > 0:
            noise_ratio = noise_analysis.get('inconsistent_regions', 0) / noise_analysis['total_regions']
        if noise_ratio > 0.2:
            enhanced_note = " (multi-scale analysis)" if noise_analysis.get('enhanced_analysis') else ""
            indicators.append(f"Noise pattern inconsistencies in {noise_analysis['inconsistent_regions']} regions{enhanced_note}")
        
        # Wavelet-based noise inconsistencies
        if noise_analysis.get('wavelet_noise_inconsistencies', 0) > 0:
            indicators.append(f"Wavelet-based noise inconsistencies detected: {noise_analysis['wavelet_noise_inconsistencies']}")
        
        # Edge discontinuities
        if edge_analysis.get('discontinuity_count', 0) > 10:
            indicators.append(f"Edge discontinuities detected: {edge_analysis['discontinuity_count']}")
        
        # Color inconsistencies
        if color_analysis.get('anomaly_count', 0) > 0:
            indicators.append(f"Color space anomalies in {color_analysis['anomaly_count']} regions")
        
        # Enhanced forensic indicators
        if self.enhanced_mode:
            # Copy-move forgery
            if copy_move_regions:
                high_conf_cm = [cm for cm in copy_move_regions if cm.confidence > 0.6]
                if high_conf_cm:
                    indicators.append(f"High-confidence copy-move forgery detected: {len(high_conf_cm)} region pairs")
                elif copy_move_regions:
                    indicators.append(f"Potential copy-move forgery: {len(copy_move_regions)} region pairs detected")
            
            # Frequency domain analysis
            if frequency_analysis:
                if len(frequency_analysis.dct_irregularities) > 0:
                    indicators.append(f"DCT coefficient irregularities: {len(frequency_analysis.dct_irregularities)} blocks")
                
                if len(frequency_analysis.wavelet_anomalies) > 0:
                    indicators.append(f"Wavelet domain anomalies: {len(frequency_analysis.wavelet_anomalies)} regions")
                
                if frequency_analysis.frequency_tampering_score > 0.5:
                    indicators.append(f"High frequency domain tampering score: {frequency_analysis.frequency_tampering_score:.3f}")
            
            # Wavelet decomposition analysis
            if wavelet_analysis.get('anomaly_regions'):
                anomaly_count = len(wavelet_analysis['anomaly_regions'])
                indicators.append(f"Wavelet decomposition anomalies: {anomaly_count} regions")
        
        # Specific high-confidence anomaly regions
        high_conf_regions = [r for r in anomaly_regions if r.confidence > 0.7]
        for region in high_conf_regions:
            indicators.append(f"High-confidence anomaly at ({region.x},{region.y}): {region.anomaly_type}")
        
        # Summary indicator
        if not indicators:
            indicators.append("No significant pixel-level anomalies detected")
        elif self.enhanced_mode:
            total_techniques = sum([
                1 if compression_analysis.get('enhanced_analysis') else 0,
                1 if noise_analysis.get('enhanced_analysis') else 0,
                1 if copy_move_regions else 0,
                1 if frequency_analysis else 0,
                1 if wavelet_analysis.get('wavelet_analysis_performed') else 0
            ])
            indicators.insert(0, f"Enhanced forensic analysis completed using {total_techniques} advanced techniques")
        
        return indicators

    def _generate_indicators(self, compression_analysis: Dict, noise_analysis: Dict,
                           edge_analysis: Dict, color_analysis: Dict,
                           anomaly_regions: List[AnomalyRegion]) -> List[str]:
        """Generate human-readable indicators (basic version)"""
        indicators = []
        
        # Compression artifacts
        if compression_analysis.get('artifact_density', 0) > 0.1:
            indicators.append(f"JPEG compression artifacts detected in {compression_analysis['artifact_count']} regions")
        
        # Noise inconsistencies
        noise_ratio = 0.0
        if noise_analysis.get('total_regions', 0) > 0:
            noise_ratio = noise_analysis.get('inconsistent_regions', 0) / noise_analysis['total_regions']
        if noise_ratio > 0.2:
            indicators.append(f"Noise pattern inconsistencies in {noise_analysis['inconsistent_regions']} regions")
        
        # Edge discontinuities
        if edge_analysis.get('discontinuity_count', 0) > 10:
            indicators.append(f"Edge discontinuities detected: {edge_analysis['discontinuity_count']}")
        
        # Color inconsistencies
        if color_analysis.get('anomaly_count', 0) > 0:
            indicators.append(f"Color space anomalies in {color_analysis['anomaly_count']} regions")
        
        # Specific anomaly regions
        for region in anomaly_regions:
            if region.confidence > 0.5:
                indicators.append(f"High-confidence anomaly at ({region.x},{region.y}): {region.anomaly_type}")
        
        if not indicators:
            indicators.append("No significant pixel-level anomalies detected")
        
        return indicators

    # ============================================================================
    # ENHANCED FORENSIC ANALYSIS METHODS
    # ============================================================================
    
    def _analyze_compression_artifacts_enhanced(self, gray_image: np.ndarray) -> Dict:
        """Enhanced JPEG compression analysis with DCT coefficient examination"""
        try:
            height, width = gray_image.shape
            artifact_map = np.zeros((height // 8, width // 8))
            artifact_count = 0
            dct_irregularities = []
            
            logger.info("Running enhanced DCT-based compression analysis")
            
            for i in range(0, height - 7, 8):
                for j in range(0, width - 7, 8):
                    block = gray_image[i:i+8, j:j+8].astype(np.float32)
                    
                    # Perform DCT analysis
                    dct_block = cv2.dct(block)
                    
                    # Analyze DCT coefficient patterns for tampering indicators
                    dct_score = self._analyze_dct_coefficients(dct_block)
                    
                    # Check for blocking artifacts at boundaries
                    boundary_score = 0.0
                    if i > 0:  # Vertical boundary
                        boundary_diff = np.abs(gray_image[i-1, j:j+8].astype(float) - 
                                             gray_image[i, j:j+8].astype(float))
                        boundary_score += np.mean(boundary_diff)
                    
                    if j > 0:  # Horizontal boundary
                        boundary_diff = np.abs(gray_image[i:i+8, j-1].astype(float) - 
                                             gray_image[i:i+8, j].astype(float))
                        boundary_score += np.mean(boundary_diff)
                    
                    # Detect quantization inconsistencies
                    quantization_score = self._detect_quantization_inconsistencies(dct_block)
                    
                    # Combined artifact score
                    combined_score = (dct_score * 0.4 + 
                                    (boundary_score / 20.0) * 0.3 + 
                                    quantization_score * 0.3)
                    
                    if combined_score > 0.5:  # Threshold for artifact detection
                        artifact_count += 1
                        artifact_map[i//8, j//8] = combined_score
                        dct_irregularities.append((i, j, combined_score))
            
            return {
                'artifact_count': artifact_count,
                'artifact_density': artifact_count / (artifact_map.size + 1e-10),
                'artifact_map': artifact_map,
                'dct_irregularities': dct_irregularities,
                'enhanced_analysis': True
            }
            
        except Exception as e:
            logger.warning(f"Error in enhanced compression analysis: {str(e)}")
            # Fallback to basic analysis
            return self._analyze_compression_artifacts_fast(gray_image)
    
    def _analyze_dct_coefficients(self, dct_block: np.ndarray) -> float:
        """Analyze DCT coefficients for tampering indicators"""
        try:
            # Check for unusual coefficient patterns
            
            # 1. High frequency coefficient analysis
            high_freq_mask = np.zeros_like(dct_block)
            high_freq_mask[4:, 4:] = 1  # High frequency region
            high_freq_energy = np.sum(np.abs(dct_block * high_freq_mask))
            total_energy = np.sum(np.abs(dct_block))
            
            if total_energy > 0:
                high_freq_ratio = high_freq_energy / total_energy
            else:
                high_freq_ratio = 0
            
            # 2. Check for periodic patterns (double compression)
            periodicity_score = self._detect_periodic_patterns(dct_block)
            
            # 3. Check coefficient magnitude distribution
            coeffs_flat = dct_block.flatten()
            coeffs_flat = coeffs_flat[coeffs_flat != 0]  # Remove zeros
            
            if len(coeffs_flat) > 5:
                # Benford's law analysis for coefficient distribution
                benford_score = self._analyze_benford_law(coeffs_flat)
            else:
                benford_score = 0
            
            # Combine scores
            dct_score = (high_freq_ratio * 0.4 + 
                        periodicity_score * 0.3 + 
                        benford_score * 0.3)
            
            return min(dct_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _detect_quantization_inconsistencies(self, dct_block: np.ndarray) -> float:
        """Detect quantization table inconsistencies"""
        try:
            # Standard JPEG quantization table (luminance)
            std_quant = np.array([
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ])
            
            # Estimate quantization table from DCT coefficients
            estimated_quant = np.zeros_like(std_quant, dtype=float)
            
            for i in range(8):
                for j in range(8):
                    if dct_block[i, j] != 0:
                        # Estimate quantization step
                        estimated_quant[i, j] = abs(dct_block[i, j])
            
            # Compare with standard quantization table
            if np.any(estimated_quant > 0):
                # Normalize for comparison
                std_norm = std_quant / np.max(std_quant)
                est_norm = estimated_quant / np.max(estimated_quant)
                
                # Calculate difference
                diff = np.mean(np.abs(std_norm - est_norm))
                return min(diff, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _detect_periodic_patterns(self, dct_block: np.ndarray) -> float:
        """Detect periodic patterns indicating double compression"""
        try:
            # Calculate autocorrelation
            dct_1d = dct_block.flatten()
            autocorr = np.correlate(dct_1d, dct_1d, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for periodic peaks
            if len(autocorr) > 8:
                peaks = []
                for i in range(2, len(autocorr) - 2):
                    if (autocorr[i] > autocorr[i-1] and 
                        autocorr[i] > autocorr[i+1] and 
                        autocorr[i] > np.mean(autocorr) * 1.2):
                        peaks.append(i)
                
                # Score based on peak periodicity
                if len(peaks) >= 2:
                    intervals = np.diff(peaks)
                    if len(intervals) > 1:
                        interval_std = np.std(intervals)
                        if interval_std < 2:  # Low variance indicates periodicity
                            return min(len(peaks) / 8.0, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_benford_law(self, coefficients: np.ndarray) -> float:
        """Analyze coefficient distribution using Benford's Law"""
        try:
            # Extract first digits
            abs_coeffs = np.abs(coefficients)
            first_digits = []
            
            for coeff in abs_coeffs:
                if coeff >= 1:
                    # Get first digit
                    while coeff >= 10:
                        coeff = coeff / 10
                    first_digits.append(int(coeff))
            
            if len(first_digits) < 10:
                return 0.0
            
            # Calculate actual distribution
            digit_counts = np.zeros(10)
            for digit in first_digits:
                if 1 <= digit <= 9:
                    digit_counts[digit] += 1
            
            # Benford's Law expected distribution
            benford_expected = np.array([0, 0.301, 0.176, 0.125, 0.097, 
                                       0.079, 0.067, 0.058, 0.051, 0.046])
            
            # Normalize actual distribution
            total_count = np.sum(digit_counts[1:])
            if total_count > 0:
                actual_dist = digit_counts / total_count
                
                # Calculate Kolmogorov-Smirnov statistic
                ks_stat = np.max(np.abs(np.cumsum(actual_dist[1:]) - 
                                       np.cumsum(benford_expected[1:])))
                
                return min(ks_stat * 2, 1.0)  # Scale to 0-1
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_noise_patterns_enhanced(self, gray_image: np.ndarray) -> Dict:
        """Enhanced noise analysis with multi-scale approach"""
        try:
            logger.info("Running enhanced multi-scale noise analysis")
            
            # Multi-scale noise analysis
            scales = [1.0, 1.5, 2.0] if self.fast_mode else [0.5, 1.0, 1.5, 2.0, 3.0]
            noise_analysis_results = {}
            
            for scale in scales:
                # Apply Gaussian filter at different scales
                filtered = gaussian_filter(gray_image, sigma=scale)
                noise = gray_image.astype(float) - filtered
                
                # Analyze noise characteristics
                noise_results = self._analyze_single_scale_noise(noise, scale)
                noise_analysis_results[f'scale_{scale}'] = noise_results
            
            # Combine results across scales
            combined_results = self._combine_multiscale_noise_results(noise_analysis_results)
            
            # Enhanced noise inconsistency detection
            enhanced_inconsistencies = self._detect_enhanced_noise_inconsistencies(gray_image)
            
            combined_results.update(enhanced_inconsistencies)
            combined_results['enhanced_analysis'] = True
            
            return combined_results
            
        except Exception as e:
            logger.warning(f"Error in enhanced noise analysis: {str(e)}")
            return self._analyze_noise_patterns_fast(gray_image)
    
    def _analyze_single_scale_noise(self, noise: np.ndarray, scale: float) -> Dict:
        """Analyze noise at a single scale"""
        h, w = noise.shape
        region_size = max(int(h // (8 / scale)), int(w // (8 / scale)), 32)
        
        noise_stats = []
        spatial_correlation = []
        
        for i in range(0, h - region_size + 1, region_size):
            for j in range(0, w - region_size + 1, region_size):
                region_noise = noise[i:i+region_size, j:j+region_size]
                
                # Basic statistics
                noise_std = np.std(region_noise)
                noise_stats.append(noise_std)
                
                # Spatial correlation
                if region_noise.size > 4:
                    autocorr = np.corrcoef(region_noise[:-1].flatten(), 
                                         region_noise[1:].flatten())[0, 1]
                    if not np.isnan(autocorr):
                        spatial_correlation.append(abs(autocorr))
        
        return {
            'noise_mean': float(np.mean(noise_stats)) if noise_stats else 0.0,
            'noise_std': float(np.std(noise_stats)) if noise_stats else 0.0,
            'spatial_correlation_mean': float(np.mean(spatial_correlation)) if spatial_correlation else 0.0,
            'total_regions': len(noise_stats)
        }
    
    def _combine_multiscale_noise_results(self, scale_results: Dict) -> Dict:
        """Combine noise analysis results across multiple scales"""
        # Extract key metrics across scales
        all_noise_means = [result['noise_mean'] for result in scale_results.values()]
        all_noise_stds = [result['noise_std'] for result in scale_results.values()]
        all_correlations = [result['spatial_correlation_mean'] for result in scale_results.values()]
        
        # Detect scale inconsistencies
        scale_inconsistency = 0
        if len(all_noise_stds) > 1:
            # Noise should decrease with increased smoothing
            for i in range(len(all_noise_stds) - 1):
                if all_noise_stds[i+1] > all_noise_stds[i] * 0.8:  # Unexpected increase
                    scale_inconsistency += 1
        
        return {
            'noise_mean': float(np.mean(all_noise_means)),
            'noise_std': float(np.std(all_noise_means)),
            'inconsistent_regions': scale_inconsistency,
            'total_regions': sum(result['total_regions'] for result in scale_results.values()),
            'scale_inconsistency_score': scale_inconsistency / max(len(all_noise_stds) - 1, 1),
            'cross_scale_correlation_variance': float(np.var(all_correlations)) if all_correlations else 0.0
        }
    
    def _detect_enhanced_noise_inconsistencies(self, gray_image: np.ndarray) -> Dict:
        """Enhanced noise inconsistency detection using advanced techniques"""
        try:
            # Wavelet-based noise analysis
            coeffs = pywt.dwt2(gray_image, 'db4')
            _, (ch, cv, cd) = coeffs
            
            # Analyze high-frequency components for noise inconsistencies
            hf_components = [ch, cv, cd]
            hf_inconsistencies = 0
            
            for component in hf_components:
                if component.size > 100:  # Ensure sufficient data
                    # Divide into regions and analyze variance
                    h, w = component.shape
                    region_size = max(h // 4, w // 4, 8)
                    
                    variances = []
                    for i in range(0, h - region_size + 1, region_size):
                        for j in range(0, w - region_size + 1, region_size):
                            region = component[i:i+region_size, j:j+region_size]
                            variances.append(np.var(region))
                    
                    if len(variances) > 1:
                        variance_std = np.std(variances)
                        variance_mean = np.mean(variances)
                        if variance_mean > 0:
                            cv = variance_std / variance_mean  # Coefficient of variation
                            if cv > 0.5:  # High variance indicates inconsistency
                                hf_inconsistencies += 1
            
            return {
                'wavelet_noise_inconsistencies': hf_inconsistencies,
                'wavelet_analysis_performed': True
            }
            
        except Exception as e:
            logger.warning(f"Enhanced noise inconsistency detection failed: {str(e)}")
            return {
                'wavelet_noise_inconsistencies': 0,
                'wavelet_analysis_performed': False
            }
    
    def _detect_copy_move_forgery(self, gray_image: np.ndarray) -> List[CopyMoveRegion]:
        """Detect copy-move forgery using SIFT/ORB feature matching"""
        copy_move_regions = []
        
        if not self.sift_detector and not self.orb_detector:
            logger.warning("No feature detectors available for copy-move detection")
            return copy_move_regions
        
        try:
            logger.info("Running copy-move forgery detection")
            
            # Use SIFT as primary, ORB as fallback
            detector = self.sift_detector if self.sift_detector else self.orb_detector
            matcher = cv2.BFMatcher()
            
            # Detect keypoints and descriptors
            keypoints, descriptors = detector.detectAndCompute(gray_image, None)
            
            if descriptors is None or len(descriptors) < 10:
                logger.info("Insufficient features for copy-move detection")
                return copy_move_regions
            
            # Match features with themselves to find duplicates
            matches = matcher.knnMatch(descriptors, descriptors, k=3)
            
            # Filter matches
            good_matches = []
            for match_group in matches:
                if len(match_group) >= 2:
                    # Exclude self-matches and apply ratio test
                    valid_matches = [m for m in match_group if m.distance > 0]
                    if len(valid_matches) >= 2:
                        m, n = valid_matches[:2]
                        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                            good_matches.append(m)
            
            if len(good_matches) < 5:
                logger.info("No significant copy-move patterns detected")
                return copy_move_regions
            
            # Group matches by spatial proximity
            copy_move_regions = self._group_copy_move_matches(good_matches, keypoints)
            
            logger.info(f"Detected {len(copy_move_regions)} potential copy-move regions")
            return copy_move_regions
            
        except Exception as e:
            logger.warning(f"Copy-move detection failed: {str(e)}")
            return copy_move_regions
    
    def _group_copy_move_matches(self, matches: List, keypoints: List) -> List[CopyMoveRegion]:
        """Group feature matches into copy-move regions"""
        copy_move_regions = []
        
        try:
            # Extract coordinate pairs
            source_points = []
            target_points = []
            
            for match in matches:
                src_pt = keypoints[match.queryIdx].pt
                dst_pt = keypoints[match.trainIdx].pt
                
                # Calculate distance between matched points
                distance = np.sqrt((src_pt[0] - dst_pt[0])**2 + (src_pt[1] - dst_pt[1])**2)
                
                # Only consider matches with sufficient distance (not same region)
                if distance > 50:  # Minimum distance threshold
                    source_points.append(src_pt)
                    target_points.append(dst_pt)
            
            if len(source_points) < 5:
                return copy_move_regions
            
            # Cluster points to find coherent regions
            source_points = np.array(source_points)
            target_points = np.array(target_points)
            
            # Simple clustering: group points within distance threshold
            distance_threshold = 100  # pixels
            used_indices = set()
            
            for i, (src_pt, tgt_pt) in enumerate(zip(source_points, target_points)):
                if i in used_indices:
                    continue
                
                # Find nearby points
                cluster_src = [src_pt]
                cluster_tgt = [tgt_pt]
                cluster_indices = [i]
                
                for j, (other_src, other_tgt) in enumerate(zip(source_points, target_points)):
                    if j in used_indices or j == i:
                        continue
                    
                    src_dist = np.linalg.norm(src_pt - other_src)
                    tgt_dist = np.linalg.norm(tgt_pt - other_tgt)
                    
                    if src_dist < distance_threshold and tgt_dist < distance_threshold:
                        cluster_src.append(other_src)
                        cluster_tgt.append(other_tgt)
                        cluster_indices.append(j)
                
                # Create copy-move region if cluster is large enough
                if len(cluster_src) >= 3:
                    used_indices.update(cluster_indices)
                    
                    # Calculate bounding boxes
                    src_array = np.array(cluster_src)
                    tgt_array = np.array(cluster_tgt)
                    
                    src_min = np.min(src_array, axis=0)
                    src_max = np.max(src_array, axis=0)
                    tgt_min = np.min(tgt_array, axis=0)
                    tgt_max = np.max(tgt_array, axis=0)
                    
                    # Calculate similarity score
                    similarity_score = min(len(cluster_src) / 20.0, 1.0)  # Normalize by expected max features
                    confidence = similarity_score * 0.8  # Conservative confidence
                    
                    copy_move_regions.append(CopyMoveRegion(
                        source_x=int(src_min[0]),
                        source_y=int(src_min[1]),
                        target_x=int(tgt_min[0]),
                        target_y=int(tgt_min[1]),
                        width=int(max(src_max[0] - src_min[0], tgt_max[0] - tgt_min[0])),
                        height=int(max(src_max[1] - src_min[1], tgt_max[1] - tgt_min[1])),
                        confidence=confidence,
                        similarity_score=similarity_score,
                        feature_count=len(cluster_src)
                    ))
            
            return copy_move_regions
            
        except Exception as e:
            logger.warning(f"Error grouping copy-move matches: {str(e)}")
            return copy_move_regions
    
    def _analyze_frequency_domain(self, gray_image: np.ndarray) -> Optional[FrequencyAnalysis]:
        """Advanced frequency domain analysis for tampering detection"""
        try:
            logger.info("Running frequency domain analysis")
            
            # FFT analysis
            fft_image = np.fft.fft2(gray_image)
            fft_magnitude = np.abs(fft_image)
            fft_phase = np.angle(fft_image)
            
            # Power spectrum analysis
            power_spectrum = np.abs(fft_image) ** 2
            power_spectrum_1d = np.mean(power_spectrum, axis=0)
            
            # Detect power spectrum anomalies
            ps_anomalies = self._detect_power_spectrum_anomalies(power_spectrum_1d)
            
            # DCT analysis for 8x8 blocks
            dct_irregularities = self._comprehensive_dct_analysis(gray_image)
            
            # Wavelet analysis
            wavelet_anomalies = self._comprehensive_wavelet_analysis(gray_image)
            
            # Calculate overall frequency tampering score
            freq_score = (
                len(dct_irregularities) / max(gray_image.size / 64, 1) * 0.4 +  # DCT score
                len(wavelet_anomalies) / max(gray_image.size / 100, 1) * 0.3 +  # Wavelet score
                min(len(ps_anomalies) / 10.0, 1.0) * 0.3  # Power spectrum score
            )
            
            return FrequencyAnalysis(
                dct_irregularities=dct_irregularities,
                wavelet_anomalies=wavelet_anomalies,
                power_spectrum_anomalies=ps_anomalies,
                frequency_tampering_score=min(freq_score, 1.0)
            )
            
        except Exception as e:
            logger.warning(f"Frequency domain analysis failed: {str(e)}")
            return None
    
    def _detect_power_spectrum_anomalies(self, power_spectrum_1d: np.ndarray) -> List[float]:
        """Detect anomalies in power spectrum"""
        anomalies = []
        
        try:
            # Smooth the power spectrum
            smoothed = gaussian_filter(power_spectrum_1d, sigma=2)
            
            # Calculate residuals
            residuals = power_spectrum_1d - smoothed
            
            # Find peaks in residuals (anomalies)
            threshold = np.std(residuals) * 2.5
            
            for i, residual in enumerate(residuals):
                if abs(residual) > threshold:
                    anomalies.append(float(residual))
            
            return anomalies
            
        except Exception:
            return anomalies
    
    def _comprehensive_dct_analysis(self, gray_image: np.ndarray) -> List[Tuple[int, int, float]]:
        """Comprehensive DCT analysis for tampering detection"""
        dct_irregularities = []
        
        try:
            height, width = gray_image.shape
            
            for i in range(0, height - 7, 8):
                for j in range(0, width - 7, 8):
                    block = gray_image[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Multiple DCT-based tests
                    irregularity_score = 0.0
                    
                    # Test 1: High frequency energy
                    hf_energy = np.sum(np.abs(dct_block[4:, 4:]))
                    total_energy = np.sum(np.abs(dct_block))
                    if total_energy > 0:
                        hf_ratio = hf_energy / total_energy
                        if hf_ratio > 0.3:  # Unusual high frequency content
                            irregularity_score += 0.3
                    
                    # Test 2: Zero coefficient patterns
                    zero_count = np.sum(dct_block == 0)
                    if zero_count < 10 or zero_count > 50:  # Unusual zero patterns
                        irregularity_score += 0.2
                    
                    # Test 3: Coefficient magnitude distribution
                    coeffs_flat = dct_block.flatten()
                    if len(coeffs_flat) > 10:
                        coeff_std = np.std(coeffs_flat)
                        coeff_mean = np.mean(np.abs(coeffs_flat))
                        if coeff_mean > 0:
                            cv = coeff_std / coeff_mean
                            if cv > 2.0:  # High variability
                                irregularity_score += 0.2
                    
                    # Test 4: AC coefficient patterns
                    ac_coeffs = dct_block.copy()
                    ac_coeffs[0, 0] = 0  # Remove DC component
                    ac_energy = np.sum(ac_coeffs ** 2)
                    if ac_energy > 1000:  # High AC energy (potential tampering)
                        irregularity_score += 0.3
                    
                    if irregularity_score > 0.4:
                        dct_irregularities.append((i, j, irregularity_score))
            
            return dct_irregularities
            
        except Exception as e:
            logger.warning(f"DCT analysis failed: {str(e)}")
            return dct_irregularities
    
    def _comprehensive_wavelet_analysis(self, gray_image: np.ndarray) -> List[Tuple[int, int, str, float]]:
        """Comprehensive wavelet analysis for tampering detection"""
        wavelet_anomalies = []
        
        try:
            # Multi-level wavelet decomposition
            wavelets = ['db4', 'haar', 'bior2.2'] if not self.fast_mode else ['db4']
            
            for wavelet_name in wavelets:
                try:
                    # Perform 2-level decomposition
                    coeffs = pywt.wavedec2(gray_image, wavelet_name, level=2)
                    
                    # Analyze each level
                    for level, coeff_tuple in enumerate(coeffs[1:], 1):  # Skip approximation
                        if isinstance(coeff_tuple, tuple) and len(coeff_tuple) == 3:
                            ch, cv, cd = coeff_tuple  # Horizontal, Vertical, Diagonal
                            
                            # Analyze each detail coefficient
                            for detail_name, detail_coeffs in [('H', ch), ('V', cv), ('D', cd)]:
                                anomaly_score = self._analyze_wavelet_coefficients(detail_coeffs)
                                
                                if anomaly_score > 0.5:
                                    # Find location of anomaly (simplified)
                                    h, w = detail_coeffs.shape
                                    max_pos = np.unravel_index(np.argmax(np.abs(detail_coeffs)), 
                                                             detail_coeffs.shape)
                                    
                                    # Scale coordinates back to original image
                                    scale_factor = 2 ** level
                                    x = max_pos[1] * scale_factor
                                    y = max_pos[0] * scale_factor
                                    
                                    wavelet_anomalies.append((
                                        int(x), int(y), 
                                        f"{wavelet_name}_{detail_name}_L{level}",
                                        anomaly_score
                                    ))
                
                except Exception as e:
                    logger.warning(f"Wavelet {wavelet_name} analysis failed: {str(e)}")
                    continue
            
            return wavelet_anomalies
            
        except Exception as e:
            logger.warning(f"Wavelet analysis failed: {str(e)}")
            return wavelet_anomalies
    
    def _analyze_wavelet_coefficients(self, coeffs: np.ndarray) -> float:
        """Analyze wavelet coefficients for anomalies"""
        try:
            if coeffs.size == 0:
                return 0.0
            
            # Statistical analysis
            coeff_mean = np.mean(np.abs(coeffs))
            coeff_std = np.std(coeffs)
            coeff_max = np.max(np.abs(coeffs))
            
            anomaly_score = 0.0
            
            # Test 1: High energy in detail coefficients
            if coeff_mean > np.std(coeffs) * 2:
                anomaly_score += 0.3
            
            # Test 2: Unusual coefficient distribution
            if coeff_std > 0:
                cv = coeff_std / (coeff_mean + 1e-10)
                if cv > 3.0:  # High coefficient of variation
                    anomaly_score += 0.3
            
            # Test 3: Extreme values
            threshold = np.mean(np.abs(coeffs)) + 3 * np.std(np.abs(coeffs))
            extreme_count = np.sum(np.abs(coeffs) > threshold)
            if extreme_count > coeffs.size * 0.05:  # More than 5% extreme values
                anomaly_score += 0.4
            
            return min(anomaly_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_wavelet_decomposition(self, gray_image: np.ndarray) -> Dict:
        """Analyze wavelet decomposition for tampering indicators"""
        try:
            result = {
                'wavelet_analysis_performed': True,
                'wavelets_tested': [],
                'anomaly_regions': [],
                'decomposition_scores': {}
            }
            
            wavelets = ['db4', 'db8', 'bior2.2', 'coif2'] if not self.fast_mode else ['db4']
            
            for wavelet_name in wavelets:
                try:
                    # Single level decomposition for analysis
                    coeffs = pywt.dwt2(gray_image, wavelet_name)
                    cA, (cH, cV, cD) = coeffs
                    
                    result['wavelets_tested'].append(wavelet_name)
                    
                    # Analyze energy distribution
                    approx_energy = np.sum(cA ** 2)
                    detail_energy = np.sum(cH ** 2) + np.sum(cV ** 2) + np.sum(cD ** 2)
                    
                    total_energy = approx_energy + detail_energy
                    if total_energy > 0:
                        detail_ratio = detail_energy / total_energy
                        result['decomposition_scores'][wavelet_name] = {
                            'detail_energy_ratio': float(detail_ratio),
                            'approximation_energy': float(approx_energy),
                            'detail_energy': float(detail_energy)
                        }
                        
                        # High detail energy might indicate tampering
                        if detail_ratio > 0.4:  # Threshold
                            result['anomaly_regions'].append({
                                'wavelet': wavelet_name,
                                'anomaly_type': 'high_detail_energy',
                                'score': float(detail_ratio)
                            })
                
                except Exception as e:
                    logger.warning(f"Wavelet {wavelet_name} decomposition failed: {str(e)}")
                    continue
            
            return result
            
        except Exception as e:
            logger.warning(f"Wavelet decomposition analysis failed: {str(e)}")
            return {'wavelet_analysis_performed': False, 'error': str(e)}


def test_pixel_analyzer():
    """Test function for the enhanced pixel forensics analyzer"""
    print("Testing Pixel Forensics Analyzer Modes...")
    
    # Test disabled mode
    print("\n1. Testing Disabled Mode...")
    try:
        analyzer_disabled = PixelForensicsAnalyzer(block_size=16, fast_mode=True, enhanced_mode="disabled")
        print(f"   ✓ Disabled mode initialized: {analyzer_disabled.enhanced_mode_type}")
        print(f"   ✓ Sample ratio: {analyzer_disabled.sample_ratio}")
        print(f"   ✓ Enhanced features: {analyzer_disabled.enhanced_mode}")
    except Exception as e:
        print(f"   ✗ Disabled mode failed: {e}")
    
    # Test basic fast mode
    print("\n2. Testing Basic Fast Mode...")
    try:
        analyzer_basic = PixelForensicsAnalyzer(block_size=16, fast_mode=True, enhanced_mode="basic_fast")
        print(f"   ✓ Basic fast mode initialized: {analyzer_basic.enhanced_mode_type}")
        print(f"   ✓ Sample ratio: {analyzer_basic.sample_ratio}")
        print(f"   ✓ Enhanced features: {analyzer_basic.enhanced_mode}")
        print(f"   ✓ Anomaly threshold: {analyzer_basic.anomaly_threshold}")
    except Exception as e:
        print(f"   ✗ Basic fast mode failed: {e}")
    
    # Test enhanced fast mode
    print("\n3. Testing Enhanced Fast Mode...")
    try:
        analyzer_enhanced_fast = PixelForensicsAnalyzer(block_size=16, fast_mode=True, enhanced_mode="enhanced_fast")
        print(f"   ✓ Enhanced fast mode initialized: {analyzer_enhanced_fast.enhanced_mode_type}")
        print(f"   ✓ Sample ratio: {analyzer_enhanced_fast.sample_ratio}")
        print(f"   ✓ Enhanced features: {analyzer_enhanced_fast.enhanced_mode}")
        print(f"   ✓ Anomaly threshold: {analyzer_enhanced_fast.anomaly_threshold}")
        print(f"   ✓ SIFT detector: {'Available' if analyzer_enhanced_fast.sift_detector else 'Not available'}")
    except Exception as e:
        print(f"   ✗ Enhanced fast mode failed: {e}")
    
    # Test enhanced thorough mode
    print("\n4. Testing Enhanced Thorough Mode...")
    try:
        analyzer_thorough = PixelForensicsAnalyzer(block_size=8, fast_mode=False, enhanced_mode="enhanced_thorough")
        print(f"   ✓ Enhanced thorough mode initialized: {analyzer_thorough.enhanced_mode_type}")
        print(f"   ✓ Sample ratio: {analyzer_thorough.sample_ratio}")
        print(f"   ✓ Enhanced features: {analyzer_thorough.enhanced_mode}")
        print(f"   ✓ Anomaly threshold: {analyzer_thorough.anomaly_threshold}")
    except Exception as e:
        print(f"   ✗ Enhanced thorough mode failed: {e}")
    
    print("\n✅ Pixel forensics analyzer test complete!")
    print("\nMode Summary:")
    print("• Disabled: Minimal analysis, 10% sampling")
    print("• Basic Fast: Standard analysis, 30% sampling, no enhanced features")
    print("• Enhanced Fast: Advanced techniques, 50% sampling, higher anomaly threshold")
    print("• Enhanced Thorough: Full analysis, 80% sampling, standard threshold")


if __name__ == "__main__":
    test_pixel_analyzer() 