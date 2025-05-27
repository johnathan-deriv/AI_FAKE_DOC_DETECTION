"""
Document Forgery Detection System - Streamlit Web Application
Modern web interface for document forgery detection with visualization
"""

import streamlit as st
import tempfile
import os
import sys
import time
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import base64
from pathlib import Path

# Try to import Plotly with fallback
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Interactive visualization will fall back to static mode.")

# Handle imports for both direct execution and module usage
try:
    # Try relative imports first (when used as module)
    from .detector import DocumentForgeryDetector
    from .visualization import DocumentVisualizer
    from .models import DocumentClassificationResult
except ImportError:
    # Fall back to absolute imports (when run directly)
    # Add the parent directory to the path so we can import from src
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from src.detector import DocumentForgeryDetector
    from src.visualization import DocumentVisualizer
    from src.models import DocumentClassificationResult

# Load environment variables for model names
OPENAI_MODEL_NAME1 = os.getenv('OPENAI_MODEL_NAME1', 'o3')  # Reasoning model
OPENAI_MODEL_NAME2 = os.getenv('OPENAI_MODEL_NAME2', 'gpt-4.1')   # Standard model

# Page configuration
st.set_page_config(
    layout="wide", 
    page_title="Document Forgery Detection System",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .status-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

class DocumentForgeryApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.detector = None
        self.visualizer = DocumentVisualizer()
        self.initialize_detector()

    def initialize_detector(self):
        """Initialize the document detector with caching"""
        try:
            if self.detector is None:
                with st.spinner("Initializing document detector..."):
                    # Use default configuration for initialization
                    self.detector = DocumentForgeryDetector(
                        vision_model_type="standard",
                        text_model_type="reasoning",
                        enable_pixel_analysis=True,
                        enable_individual_agents=False,  # Will be set dynamically
                        pixel_block_size=16,
                        fast_pixel_mode=True
                    )
                st.success("✅ Document detector initialized successfully")
        except Exception as e:
            st.error(f"❌ Error initializing detector: {str(e)}")
            self.detector = None

    def get_or_create_detector(self, config: Dict) -> DocumentForgeryDetector:
        """Get detector with current configuration or create new one if needed"""
        try:
            # Determine pixel analysis settings based on mode
            pixel_mode = config["pixel_forensics_mode"]
            
            if pixel_mode == "disabled":
                enable_pixel_analysis = False
                enhanced_pixel_forensics = "disabled"
            elif pixel_mode == "basic_fast":
                enable_pixel_analysis = True
                enhanced_pixel_forensics = "basic_fast"  # Pass the specific mode
            elif pixel_mode in ["enhanced_fast", "enhanced_thorough"]:
                enable_pixel_analysis = True
                enhanced_pixel_forensics = pixel_mode  # Pass the specific mode
            else:
                # Default fallback
                enable_pixel_analysis = True
                enhanced_pixel_forensics = "basic_fast"
            
            # Map model names to model types
            vision_model_type = self._get_model_type_from_model(config["vision_model"])
            text_model_type = self._get_model_type_from_model(config["text_model"])
            
            # Create a new detector with current configuration
            detector = DocumentForgeryDetector(
                vision_model_type=vision_model_type,
                text_model_type=text_model_type,
                enable_pixel_analysis=enable_pixel_analysis,
                enable_individual_agents=config["enable_individual_agents"],
                pixel_block_size=config["pixel_block_size"],
                fast_pixel_mode=config["fast_pixel_mode"],
                enhanced_pixel_forensics=enhanced_pixel_forensics
            )
            return detector
        except Exception as e:
            st.error(f"❌ Error creating detector with configuration: {str(e)}")
            return self.detector  # Fall back to default detector

    def _get_model_type_from_model(self, model_name: str) -> str:
        """Map model name to model type"""
        if model_name == OPENAI_MODEL_NAME1 or "o1" in model_name.lower() or "o4" in model_name.lower():
            return "reasoning"
        elif model_name == OPENAI_MODEL_NAME2 or "gpt-4" in model_name.lower():
            return "standard"
        else:
            # Default fallback
            return "standard"

    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.title("🔧 Configuration")
        
        # OpenAI Model Selection
        st.sidebar.subheader("🤖 OpenAI Model Settings")
        
        vision_model = st.sidebar.selectbox(
            "Vision Model",
            [OPENAI_MODEL_NAME2, OPENAI_MODEL_NAME1],  # Standard first as default
            help="Choose the OpenAI model for vision analysis",
            format_func=lambda x: {
                OPENAI_MODEL_NAME1: f"🧠 {OPENAI_MODEL_NAME1} (Reasoning - Temp=1)",
                OPENAI_MODEL_NAME2: f"⚡ {OPENAI_MODEL_NAME2} (Standard - Temp=0)"
            }.get(x, x)
        )
        
        text_model = st.sidebar.selectbox(
            "Text Model", 
            [OPENAI_MODEL_NAME1, OPENAI_MODEL_NAME2],  # Reasoning first as default for text
            help="Choose the OpenAI model for text analysis",
            format_func=lambda x: {
                OPENAI_MODEL_NAME1: f"🧠 {OPENAI_MODEL_NAME1} (Reasoning - Temp=1)",
                OPENAI_MODEL_NAME2: f"⚡ {OPENAI_MODEL_NAME2} (Standard - Temp=0)"
            }.get(x, x)
        )
        
        # Show model information
        if vision_model == OPENAI_MODEL_NAME1 or text_model == OPENAI_MODEL_NAME1:
            st.sidebar.info("🧠 **Reasoning Models**: Use temperature=1 and are optimized for complex reasoning tasks")
        if vision_model == OPENAI_MODEL_NAME2 or text_model == OPENAI_MODEL_NAME2:
            st.sidebar.info("⚡ **Standard Models**: Use temperature=0 for deterministic, fast responses")
        
        # Pixel Forensics Options
        st.sidebar.subheader("🔬 Pixel Forensics")
        pixel_forensics_mode = st.sidebar.selectbox(
            "Operation Mode",
            ["disabled", "basic_fast", "enhanced_fast", "enhanced_thorough"],
            index=2,  # Default to enhanced_fast
            help="Choose the level of pixel forensics analysis",
            format_func=lambda x: {
                "disabled": "🚫 Disabled (Fastest)",
                "basic_fast": "⚡ Basic Fast (Standard)",
                "enhanced_fast": "🔍 Enhanced Fast (Recommended)", 
                "enhanced_thorough": "🎯 Enhanced Thorough (Most Comprehensive)"
            }.get(x, x)
        )
        
        # Show detailed description based on selected mode
        if pixel_forensics_mode == "disabled":
            st.sidebar.info("🚫 **Disabled**: No pixel-level analysis. Fastest processing time.")
        elif pixel_forensics_mode == "basic_fast":
            st.sidebar.info("⚡ **Basic Fast**: Standard compression, noise, and edge analysis.")
        elif pixel_forensics_mode == "enhanced_fast":
            st.sidebar.success("🔍 **Enhanced Fast**: DCT analysis, basic wavelets, copy-move detection. Optimized for web apps.")
        elif pixel_forensics_mode == "enhanced_thorough":
            st.sidebar.warning("🎯 **Enhanced Thorough**: Full analysis with all techniques. May take longer.")
        
        # Additional pixel forensics settings for enhanced modes
        pixel_block_size = 16
        fast_pixel_mode = True
        
        if pixel_forensics_mode in ["enhanced_fast", "enhanced_thorough"]:
            st.sidebar.subheader("Advanced Settings")
            pixel_block_size = st.sidebar.selectbox(
                "Block Size",
                [8, 16, 32],
                index=1,  # Default to 16
                help="Size of blocks for pixel analysis (smaller = more detailed but slower)"
            )
            
            if pixel_forensics_mode == "enhanced_thorough":
                fast_pixel_mode = st.sidebar.checkbox(
                    "Fast Mode Optimizations",
                    value=False,
                    help="Disable for maximum accuracy in thorough mode"
                )
        
        # Analysis Options
        st.sidebar.subheader("Analysis Options")
        enable_individual_agents = st.sidebar.checkbox(
            "Enable Individual Agents",
            value=False,
            help="Run specialized analysis agents for different forgery types (font, layout, content, etc.)"
        )
        
        enable_visualization = st.sidebar.checkbox(
            "Enable Visualization",
            value=True,
            help="Generate annotated images for detected forgery indicators"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence score for flagging as suspicious"
        )
        
        # PDF Options
        st.sidebar.subheader("PDF Options")
        max_pages = st.sidebar.number_input(
            "Max Pages to Analyze",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of PDF pages to analyze"
        )
        
        # Show individual agents info if enabled
        if enable_individual_agents:
            st.sidebar.info("🤖 Individual Agents will analyze: Digital Artifacts, Font Inconsistencies, Date Verification, Layout Issues, Design Verification, and Content Analysis")
        
        return {
            "vision_model": vision_model,
            "text_model": text_model,
            "pixel_forensics_mode": pixel_forensics_mode,
            "pixel_block_size": pixel_block_size,
            "fast_pixel_mode": fast_pixel_mode,
            "enable_individual_agents": enable_individual_agents,
            "enable_visualization": enable_visualization,
            "confidence_threshold": confidence_threshold,
            "max_pages": max_pages
        }

    def render_file_upload(self):
        """Render the file upload section"""
        st.markdown('<h2 class="main-header">🔍 Document Forgery Detection System</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Upload any document (Image or PDF) to check its legitimacy and visualize potential 
        forgery indicators. Supports financial documents, medical bills, invoices, receipts, and more.
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["jpg", "jpeg", "png", "pdf", "bmp", "tiff", "webp"],
            help="Upload an image (JPG, PNG, etc.) or a PDF document."
        )
        
        return uploaded_file

    def render_document_preview(self, uploaded_file):
        """Render document preview"""
        try:
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption=f'📄 {uploaded_file.name}', use_container_width=True)
            elif uploaded_file.type == "application/pdf":
                # Save uploaded PDF to temporary file for pdf2image
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Import pdf2image (already used elsewhere in the codebase)
                    from pdf2image import convert_from_path
                    
                    # Convert PDF pages to images
                    st.info(f"📄 PDF Preview: {uploaded_file.name}")
                    
                    # Convert first few pages (limit to 3 for preview)
                    pdf_images = convert_from_path(tmp_file_path, dpi=150, first_page=1, last_page=3)
                    
                    if pdf_images:
                        for i, img in enumerate(pdf_images):
                            st.image(
                                img, 
                                caption=f'Page {i+1}', 
                                use_container_width=True
                            )
                        
                        if len(pdf_images) == 3:
                            st.caption("📝 Preview limited to first 3 pages")
                    else:
                        st.warning("⚠️ Could not extract pages from PDF for preview.")
                        
                except Exception as pdf_error:
                    st.warning(f"⚠️ Could not generate PDF preview: {str(pdf_error)}")
                    st.info(f"📄 PDF Uploaded: {uploaded_file.name}")
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            else:
                st.warning("⚠️ Unsupported file type for preview.")
        except Exception as e:
            st.error(f"❌ Error generating preview: {str(e)}")

    def analyze_document(self, temp_file_path: str, config: Dict) -> Dict[str, Any]:
        """Analyze the uploaded document"""
        detector = self.get_or_create_detector(config)
        if not detector:
            st.error("❌ Detector not initialized. Please refresh the page.")
            return None
        
        try:
            start_time = time.time()
            
            # Run the analysis
            with st.spinner("🕵️ Analyzing document... This may take a moment."):
                results = detector.analyze_document(
                    temp_file_path,
                    save_report=config["enable_visualization"],
                    max_pages=config["max_pages"]
                )
            
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            return results
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            return None

    def render_analysis_results(self, results: Dict[str, Any], config: Dict):
        """Render the analysis results"""
        if not results:
            return
        
        # Main verdict
        verdict = results.get("verdict", "unknown")
        overall_score = results.get("overall_score", 0.0)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if verdict.lower() == "suspicious" or overall_score > config["confidence_threshold"]:
                st.markdown("""
                <div class="status-danger">
                    🚨 <strong>Document Classified as SUSPICIOUS/FAKE</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-success">
                    ✅ <strong>Document Classified as LEGITIMATE</strong>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Overall Score", f"{overall_score:.3f}")
        
        with col3:
            processing_time = results.get("processing_time", 0)
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        # Detailed reasoning - generate from recommendations and indicators
        recommendations = results.get("recommendations", [])
        indicators = results.get("indicators", [])
        
        # Create comprehensive reasoning text
        reasoning_parts = []
        
        if verdict.lower() == "suspicious":
            reasoning_parts.append(f"🚨 **Analysis Conclusion**: Document flagged as SUSPICIOUS with confidence score {overall_score:.3f}")
        else:
            reasoning_parts.append(f"✅ **Analysis Conclusion**: Document appears LEGITIMATE with confidence score {overall_score:.3f}")
        
        if indicators:
            reasoning_parts.append(f"\n**Evidence Found** ({len(indicators)} indicators):")
            for i, indicator in enumerate(indicators[:5], 1):  # Show top 5 indicators
                if isinstance(indicator, dict):
                    desc = indicator.get('description', str(indicator))
                    source = indicator.get('source', 'unknown')
                    severity = indicator.get('severity', 'medium')
                    reasoning_parts.append(f"{i}. [{source.upper()} - {severity}] {desc}")
                else:
                    reasoning_parts.append(f"{i}. {str(indicator)}")
            
            if len(indicators) > 5:
                reasoning_parts.append(f"... and {len(indicators) - 5} more indicators (see detailed analysis below)")
        
        if recommendations:
            reasoning_parts.append(f"\n**Recommendations**:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                reasoning_parts.append(f"• {rec}")
        
        if not indicators and not recommendations:
            reasoning_parts.append("\nNo specific security concerns identified in the analysis.")
        
        reasoning = "\n".join(reasoning_parts)
        
        st.subheader("📋 Analysis Summary")
        st.write(reasoning)
        
        # Module results
        self.render_module_results(results)
        
        # Visualization if available
        if config["enable_visualization"]:
            self.render_visualization(results)

    def render_module_results(self, results: Dict[str, Any]):
        """Render individual module results"""
        st.subheader("🔬 Detailed Module Analysis")
        
        # Get detailed results from fusion analyzer output
        detailed_results = results.get("detailed_results", {})
        module_scores = results.get("module_scores", {})
        
        if detailed_results or module_scores:
            # Create mapping from detailed results and module scores
            module_results = {}
            
            # Map detailed results to module results format
            if detailed_results.get("metadata_analysis"):
                module_results["Metadata Analysis"] = detailed_results["metadata_analysis"]
            
            if detailed_results.get("vision_analysis"):
                module_results["Vision Forensics"] = detailed_results["vision_analysis"]
            
            if detailed_results.get("layout_analysis"):
                module_results["Layout Reasoning"] = detailed_results["layout_analysis"]
            
            if detailed_results.get("pixel_analysis"):
                module_results["Pixel Forensics"] = detailed_results["pixel_analysis"]
            
            if detailed_results.get("individual_agents_analysis"):
                module_results["Individual Agents"] = detailed_results["individual_agents_analysis"]
            
            # Add module scores information if available
            for module_name, score_data in module_scores.items():
                if score_data.get('available', False):
                    display_name = {
                        'metadata': 'Metadata Analysis',
                        'vision': 'Vision Forensics', 
                        'layout': 'Layout Reasoning',
                        'pixel': 'Pixel Forensics',
                        'individual_agents': 'Individual Agents'
                    }.get(module_name, module_name.title())
                    
                    if display_name not in module_results:
                        module_results[display_name] = score_data
            
            if module_results:
                tabs = st.tabs(list(module_results.keys()))
                
                for tab, (module_name, module_result) in zip(tabs, module_results.items()):
                    with tab:
                        if module_name == "Individual Agents":
                            self.render_individual_agents_result(module_result)
                        else:
                            self.render_single_module_result(module_name, module_result)
            else:
                st.info("ℹ️ No detailed module results available.")
        else:
            st.info("ℹ️ No detailed module results available.")

    def render_individual_agents_result(self, result: Dict):
        """Render individual agents results with detailed breakdown"""
        # Main summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = result.get("score", 0.0)
            st.metric("Aggregate Score", f"{score:.3f}")
        
        with col2:
            suspicious = result.get("suspicious", False)
            status = "🔴 Suspicious" if suspicious else "🟢 Clean"
            st.write(f"**Status:** {status}")
        
        with col3:
            details = result.get("details", {})
            total_agents = details.get("total_agents", 0)
            suspicious_count = details.get("suspicious_agent_count", 0)
            st.metric("Suspicious Agents", f"{suspicious_count}/{total_agents}")
        
        # Individual agent results
        agent_results = result.get("agent_results", {})
        if agent_results:
            st.write("### Individual Agent Results")
            
            # Create expandable sections for each agent
            for agent_name, agent_result in agent_results.items():
                agent_display_name = agent_name.replace('_', ' ').title()
                agent_score = agent_result.get('score', 0.0)
                agent_suspicious = agent_result.get('suspicious', False)
                
                # Use colored emoji based on suspicion
                status_emoji = "🔴" if agent_suspicious else "🟢"
                
                with st.expander(f"{status_emoji} {agent_display_name} (Score: {agent_score:.3f})"):
                    # Agent indicators
                    indicators = agent_result.get('indicators', [])
                    if indicators:
                        st.write(f"**Found {len(indicators)} indicators:**")
                        for indicator in indicators:
                            st.write(f"• {indicator}")
                    else:
                        st.write("No specific indicators found by this agent.")
                    
                    # Agent reasoning
                    reasoning = agent_result.get('reasoning', '')
                    if reasoning:
                        st.write("**Analysis Details:**")
                        # Truncate very long reasoning
                        if len(reasoning) > 500:
                            st.write(reasoning[:500] + "...")
                            if st.button(f"Show full reasoning", key=f"show_{agent_name}"):
                                st.write(reasoning)
                        else:
                            st.write(reasoning)
        
        # Overall indicators
        overall_indicators = result.get("indicators", [])
        if overall_indicators:
            st.write("### Overall Indicators")
            for indicator in overall_indicators:
                st.write(f"• {indicator}")

    def render_single_module_result(self, module_name: str, result: Dict):
        """Render a single module's results"""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            score = result.get("score", 0.0)
            suspicious = result.get("suspicious", False)
            
            st.metric(f"{module_name.title()} Score", f"{score:.3f}")
            
            status = "🔴 Suspicious" if suspicious else "🟢 Clean"
            st.write(f"**Status:** {status}")
        
        with col2:
            indicators = result.get("indicators", [])
            if indicators:
                st.write(f"**Found {len(indicators)} indicators:**")
                for indicator in indicators:
                    st.write(f"• {indicator}")
            else:
                st.write("**No specific indicators found**")
        
        # Error information if present
        if "error" in result:
            st.warning(f"⚠️ Module error: {result['error']}")

    def render_visualization(self, results: Dict[str, Any]):
        """Render visualization results"""
        st.subheader("🎨 Forgery Visualization")
        
        # Add info about interactive mode
        st.info("💡 **New Feature**: Interactive mode shows detailed information when you hover over detected regions on the document!")
        
        # Check for annotated images
        report = results.get("report", {})
        
        if "annotated_images" in report and report["annotated_images"]:
            st.info("📑 Multi-page document visualization:")
            
            # Add toggle for interactive vs static view
            view_mode = st.radio(
                "Visualization Mode:",
                ["Interactive (with hover details)", "Static (original)"],
                key="viz_mode"
            )
            
            if view_mode.startswith("Interactive"):
                self.render_interactive_visualization(results)
            else:
                for i, img_data in enumerate(report["annotated_images"]):
                    if img_data is not None:
                        # Convert BGR to RGB for display
                        img_rgb = self.visualizer.convert_to_rgb(img_data)
                        st.image(
                            img_rgb, 
                            caption=f'Annotated Forgery Indicators - Page {i+1}',
                            use_container_width=True
                        )
        
        elif "annotated_image" in report and report["annotated_image"] is not None:
            st.info("📄 Single page document visualization:")
            
            # Add toggle for interactive vs static view
            view_mode = st.radio(
                "Visualization Mode:",
                ["Interactive (with hover details)", "Static (original)"],
                key="viz_mode_single"
            )
            
            if view_mode.startswith("Interactive"):
                self.render_interactive_visualization(results)
            else:
                img_rgb = self.visualizer.convert_to_rgb(report["annotated_image"])
                st.image(
                    img_rgb,
                    caption='Annotated Forgery Indicators',
                    use_container_width=True
                )
        
        else:
            # Fallback visualization
            st.info("ℹ️ No specific indicator regions found for visualization.")
            
            # Show legend if indicators were detected
            indicators = results.get("indicators", [])
            if indicators:
                # Handle both string indicators and dict indicators
                detected_types = set()
                for indicator in indicators:
                    if isinstance(indicator, str):
                        detected_types.add(indicator)
                    elif isinstance(indicator, dict):
                        # Extract indicator type from dict (assuming it has a 'type' or similar key)
                        indicator_type = indicator.get('type') or indicator.get('name') or str(indicator)
                        detected_types.add(indicator_type)
                    else:
                        # Convert other types to string
                        detected_types.add(str(indicator))
                
                legend_html = self.visualizer.create_legend_html(detected_types)
                
                if legend_html:
                    st.subheader("🏷️ Legend")
                    st.markdown(legend_html, unsafe_allow_html=True)

    def render_interactive_visualization(self, results: Dict[str, Any]):
        """Render interactive visualization with hover details using Plotly"""
        if not PLOTLY_AVAILABLE:
            st.warning("📊 Interactive visualization requires Plotly. Showing static visualization instead.")
            # Fall back to static visualization
            report = results.get("report", {})
            if "annotated_images" in report and report["annotated_images"]:
                for i, img_data in enumerate(report["annotated_images"]):
                    if img_data is not None:
                        img_rgb = self.visualizer.convert_to_rgb(img_data)
                        st.image(img_rgb, caption=f'Annotated Forgery Indicators - Page {i+1}', use_container_width=True)
            elif "annotated_image" in report and report["annotated_image"] is not None:
                img_rgb = self.visualizer.convert_to_rgb(report["annotated_image"])
                st.image(img_rgb, caption='Annotated Forgery Indicators', use_container_width=True)
            return
        
        try:
            report = results.get("report", {})
            detailed_results = results.get("detailed_results", {})
            
            # Get annotated images
            annotated_images = report.get("annotated_images") or [report.get("annotated_image")]
            if not annotated_images or not any(img is not None for img in annotated_images):
                st.warning("No annotated images available for interactive visualization.")
                return
            
            # Extract indicators and regions data
            hover_data = self._extract_hover_data(detailed_results, results)
            
            # Create interactive visualizations for each page
            for page_idx, img_data in enumerate(annotated_images):
                if img_data is None:
                    continue
                
                page_number = page_idx + 1
                st.write(f"**Page {page_number} - Interactive Visualization**")
                
                # Convert image to RGB for display
                img_rgb = self.visualizer.convert_to_rgb(img_data)
                height, width = img_rgb.shape[:2]
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add the base image
                fig.add_layout_image(
                    dict(
                        source=Image.fromarray(img_rgb),
                        xref="x",
                        yref="y", 
                        x=0,
                        y=height,
                        sizex=width,
                        sizey=height,
                        sizing="stretch",
                        opacity=1.0,
                        layer="below"
                    )
                )
                
                # Add interactive regions for this page
                page_hover_data = hover_data.get(page_number, [])
                
                for region in page_hover_data:
                    self._add_interactive_region(fig, region, height)
                
                # Set up the layout
                fig.update_layout(
                    title=f"Document Analysis - Page {page_number} (Hover over colored regions for details)",
                    xaxis=dict(
                        range=[0, width],
                        showgrid=False,
                        showticklabels=False,
                        zeroline=False
                    ),
                    yaxis=dict(
                        range=[0, height],
                        showgrid=False,
                        showticklabels=False,
                        zeroline=False,
                        scaleanchor="x",
                        scaleratio=1
                    ),
                    width=800,
                    height=int(800 * height / width),
                    margin=dict(l=0, r=0, t=50, b=0),
                    showlegend=True
                )
                
                # Display the interactive plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Add region details below the plot
                if page_hover_data:
                    with st.expander(f"📋 Detailed Analysis - Page {page_number}"):
                        for i, region in enumerate(page_hover_data):
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.write(f"**Region {i+1}:**")
                                st.write(f"Type: {region['type']}")
                                st.write(f"Confidence: {region['confidence']:.3f}")
                            with col2:
                                st.write(f"**Description:** {region['description']}")
                                if region.get('source'):
                                    st.write(f"**Source:** {region['source']}")
                                if region.get('reasoning'):
                                    st.write(f"**Details:** {region['reasoning'][:200]}...")
                
                # Add metadata summary for the document
                self._render_metadata_summary(detailed_results, page_number)
            
        except Exception as e:
            st.error(f"❌ Error creating interactive visualization: {str(e)}")
            st.info("💡 Falling back to static visualization mode.")

    def _render_metadata_summary(self, detailed_results: Dict, page_number: int):
        """Render metadata summary with interactive insights"""
        try:
            metadata_analysis = detailed_results.get("metadata_analysis", {})
            
            if metadata_analysis and page_number == 1:  # Only show once for first page
                with st.expander(f"📊 Document Metadata Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📋 Document Properties:**")
                        details = metadata_analysis.get("details", {})
                        if details:
                            # Basic document info
                            if details.get("creation_date"):
                                st.write(f"Created: {details['creation_date']}")
                            if details.get("modification_date"):
                                st.write(f"Modified: {details['modification_date']}")
                            if details.get("creator"):
                                st.write(f"Creator: {details['creator']}")
                            if details.get("producer"):
                                st.write(f"Producer: {details['producer']}")
                            if details.get("title"):
                                st.write(f"Title: {details['title']}")
                            
                            # File properties
                            if details.get("file_size"):
                                st.write(f"File Size: {details['file_size']}")
                            if details.get("page_count"):
                                st.write(f"Pages: {details['page_count']}")
                    
                    with col2:
                        st.write("**🔍 Analysis Results:**")
                        score = metadata_analysis.get("score", 0.0)
                        suspicious = metadata_analysis.get("suspicious", False)
                        
                        status_color = "🔴" if suspicious else "🟢"
                        st.write(f"**Status:** {status_color} {'Suspicious' if suspicious else 'Clean'}")
                        st.write(f"**Score:** {score:.3f}")
                        
                        # Metadata indicators
                        indicators = metadata_analysis.get("indicators", [])
                        if indicators:
                            st.write("**Findings:**")
                            for indicator in indicators[:3]:  # Show top 3
                                st.write(f"• {indicator}")
                        else:
                            st.write("**Findings:** No metadata issues detected")
        
        except Exception as e:
            pass  # Silently fail for metadata rendering to not interrupt visualization

    def _extract_hover_data(self, detailed_results: Dict, results: Dict) -> Dict[int, List[Dict]]:
        """Extract hover data from analysis results for each page"""
        hover_data = {}
        
        try:
            # Extract pixel forensics anomaly regions
            pixel_analysis = detailed_results.get("pixel_analysis", {})
            pixel_page_results = pixel_analysis.get("page_results", [])
            
            for page_result in pixel_page_results:
                page_num = page_result.get("page_number", 1)
                if page_num not in hover_data:
                    hover_data[page_num] = []
                
                # Add anomaly regions
                anomaly_regions = page_result.get("anomaly_regions", [])
                for region in anomaly_regions:
                    hover_data[page_num].append({
                        "type": "Pixel Anomaly",
                        "subtype": region.get("type", "Unknown"),
                        "x": region.get("x", 0),
                        "y": region.get("y", 0),
                        "width": region.get("width", 50),
                        "height": region.get("height", 50),
                        "confidence": region.get("confidence", 0.0),
                        "description": region.get("description", f"Pixel anomaly detected: {region.get('type', 'unknown type')}"),
                        "source": "Pixel Forensics",
                        "reasoning": f"Anomaly type: {region.get('type', 'unknown')}, Confidence: {region.get('confidence', 0.0):.3f}",
                        "color": "rgba(255, 0, 0, 0.3)"  # Red for pixel anomalies
                    })
            
            # Extract vision analysis suspected regions
            vision_analysis = detailed_results.get("vision_analysis", {})
            vision_page_results = vision_analysis.get("page_results", [])
            
            for page_result in vision_page_results:
                page_num = page_result.get("page_number", 1)
                if page_num not in hover_data:
                    hover_data[page_num] = []
                
                # Add suspected regions
                suspected_regions = page_result.get("suspected_regions", [])
                for region in suspected_regions:
                    bbox = region.get("bbox", {})
                    if bbox:
                        hover_data[page_num].append({
                            "type": "Vision Forensics",
                            "subtype": region.get("description", "Suspicious region"),
                            "x": bbox.get("x", 0),
                            "y": bbox.get("y", 0),
                            "width": bbox.get("width", 50),
                            "height": bbox.get("height", 50),
                            "confidence": region.get("confidence", page_result.get("confidence", 0.0)),
                            "description": region.get("description", "Suspicious visual content detected"),
                            "source": "Vision Analysis",
                            "reasoning": page_result.get("reasoning", "Visual forensics analysis"),
                            "color": "rgba(0, 0, 255, 0.3)"  # Blue for vision analysis
                        })
                
                # If vision analysis found issues but no specific regions, create a general indicator
                if page_result.get("suspicious", False) and not suspected_regions:
                    indicators = page_result.get("indicators", [])
                    if indicators:
                        hover_data[page_num].append({
                            "type": "Vision Analysis",
                            "subtype": "General Suspicious Content",
                            "x": 50,
                            "y": 50,
                            "width": 150,
                            "height": 30,
                            "confidence": page_result.get("confidence", 0.0),
                            "description": f"Vision analysis detected issues: {'; '.join(indicators[:2])}",
                            "source": "Vision Analysis",
                            "reasoning": page_result.get("reasoning", "General visual analysis findings"),
                            "color": "rgba(0, 0, 255, 0.3)"
                        })
            
            # Extract individual agents results
            agents_analysis = detailed_results.get("individual_agents_analysis", {})
            agent_results = agents_analysis.get("agent_results", {})
            
            # For agents, we'll create general page-level indicators since they don't have specific coordinates
            agent_y_offset = 100  # Start lower to avoid overlapping with vision indicators
            for agent_name, agent_result in agent_results.items():
                if agent_result.get("suspicious", False):
                    indicators = agent_result.get("indicators", [])
                    reasoning = agent_result.get("reasoning", "")
                    
                    # Add to page 1 by default (agents analyze whole document)
                    if 1 not in hover_data:
                        hover_data[1] = []
                    
                    # Create a general indicator region (left side, stacked vertically)
                    hover_data[1].append({
                        "type": "Individual Agent",
                        "subtype": agent_name.replace("_", " ").title(),
                        "x": 10,
                        "y": agent_y_offset,
                        "width": 200,
                        "height": 25,
                        "confidence": agent_result.get("score", 0.0),
                        "description": f"{agent_name.replace('_', ' ').title()}: {'; '.join(indicators[:2])}",
                        "source": f"{agent_name.replace('_', ' ').title()} Agent",
                        "reasoning": reasoning[:300] if reasoning else "No detailed reasoning available",
                        "color": "rgba(255, 165, 0, 0.3)"  # Orange for agents
                    })
                    agent_y_offset += 30  # Move down for next agent
            
            # If we have general indicators but no specific regions, create summary indicators
            general_indicators = results.get("indicators", [])
            if general_indicators and not any(hover_data.values()):
                # Create general indicators for page 1
                if 1 not in hover_data:
                    hover_data[1] = []
                
                y_pos = 200
                for i, indicator in enumerate(general_indicators[:5]):  # Limit to top 5
                    if isinstance(indicator, dict):
                        desc = indicator.get('description', str(indicator))
                        source = indicator.get('source', 'Analysis')
                        severity = indicator.get('severity', 'medium')
                    else:
                        desc = str(indicator)
                        source = 'General Analysis'
                        severity = 'medium'
                    
                    color_map = {
                        'high': 'rgba(255, 0, 0, 0.4)',     # Red
                        'medium': 'rgba(255, 165, 0, 0.4)', # Orange
                        'low': 'rgba(255, 255, 0, 0.4)'     # Yellow
                    }
                    
                    hover_data[1].append({
                        "type": "General Finding",
                        "subtype": f"Indicator {i+1}",
                        "x": 20,
                        "y": y_pos + (i * 35),
                        "width": 250,
                        "height": 30,
                        "confidence": 0.5,  # Default confidence for general indicators
                        "description": desc[:100],
                        "source": source,
                        "reasoning": f"Severity: {severity}. {desc}",
                        "color": color_map.get(severity, 'rgba(128, 128, 128, 0.4)')
                    })
            
        except Exception as e:
            st.error(f"Error extracting hover data: {str(e)}")
        
        return hover_data

    def _add_interactive_region(self, fig: go.Figure, region: Dict, image_height: int):
        """Add an interactive region to the Plotly figure"""
        try:
            # Convert y-coordinate (OpenCV uses top-left origin, Plotly uses bottom-left)
            y_plotly = image_height - region["y"] - region["height"]
            
            # Create hover text
            hover_text = (
                f"<b>{region['type']}: {region['subtype']}</b><br>" +
                f"<b>Confidence:</b> {region['confidence']:.3f}<br>" +
                f"<b>Source:</b> {region['source']}<br>" +
                f"<b>Description:</b> {region['description']}<br>" +
                f"<b>Details:</b> {region['reasoning'][:100]}..."
            )
            
            # Add rectangular region
            fig.add_shape(
                type="rect",
                x0=region["x"],
                y0=y_plotly,
                x1=region["x"] + region["width"],
                y1=y_plotly + region["height"],
                fillcolor=region["color"],
                line=dict(color=region["color"].replace("0.3", "0.8"), width=2),
                opacity=0.7
            )
            
            # Add invisible scatter point for hover functionality
            fig.add_trace(go.Scatter(
                x=[region["x"] + region["width"]/2],
                y=[y_plotly + region["height"]/2],
                mode="markers",
                marker=dict(
                    size=max(region["width"], region["height"])/10,
                    opacity=0.01,  # Nearly invisible
                    color=region["color"]
                ),
                hovertext=hover_text,
                hoverinfo="text",
                name=region["subtype"],
                showlegend=True,
                legendgroup=region["type"]
            ))
            
        except Exception as e:
            st.error(f"Error adding interactive region: {str(e)}")

    def render_system_status(self):
        """Render comprehensive system status information"""
        with st.expander("🔧 System Status", expanded=False):
            if self.detector:
                status = self.detector.get_system_status()
                
                # Overall System Health Summary
                st.subheader("📊 System Health Overview")
                overall_health = status.get("system_health", {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    api_status = overall_health.get('apis_operational', False)
                    st.metric("API Services", "🟢 Online" if api_status else "🔴 Offline")
                
                with col2:
                    core_status = overall_health.get('core_modules_operational', False)
                    st.metric("Core Modules", "🟢 Ready" if core_status else "🔴 Issues")
                
                with col3:
                    deps_status = overall_health.get('dependencies_operational', False)
                    st.metric("Dependencies", "🟢 OK" if deps_status else "🔴 Missing")
                
                with col4:
                    system_status = overall_health.get('overall_operational', False)
                    st.metric("System Status", "🟢 Operational" if system_status else "🔴 Degraded")
                
                st.markdown("---")
                
                # Detailed Status by Category
                col1, col2 = st.columns(2)
                
                with col1:
                    # API Status
                    st.write("**🌐 API Services:**")
                    apis = status.get("apis", {})
                    for api_name, api_status in apis.items():
                        status_icon = "🟢" if api_status else "🔴"
                        display_name = api_name.replace('_', ' ').title()
                        st.write(f"{status_icon} {display_name}")
                    
                    # Core Analysis Modules
                    st.write("\n**🔬 Core Analysis Modules:**")
                    core_modules = status.get("core_modules", {})
                    for module_name, module_status in core_modules.items():
                        status_icon = "🟢" if module_status else "🔴"
                        display_name = module_name.replace('_', ' ').title()
                        st.write(f"{status_icon} {display_name}")
                    
                    # Optional Enhancement Modules
                    st.write("\n**⚡ Enhancement Modules:**")
                    optional_modules = status.get("optional_modules", {})
                    for module_name, module_status in optional_modules.items():
                        status_icon = "🟢" if module_status else "🟡"
                        display_name = module_name.replace('_', ' ').title()
                        st.write(f"{status_icon} {display_name}")
                
                with col2:
                    # System Dependencies
                    st.write("**📦 System Dependencies:**")
                    dependencies = status.get("dependencies", {})
                    for dep_name, dep_status in dependencies.items():
                        status_icon = "🟢" if dep_status else "🔴"
                        display_name = dep_name.replace('_', ' ').title()
                        st.write(f"{status_icon} {display_name}")
                    
                    # System Resources
                    st.write("\n**💾 System Resources:**")
                    resources = status.get("system_resources", {})
                    for resource_name, resource_status in resources.items():
                        if isinstance(resource_status, bool):
                            status_icon = "🟢" if resource_status else "🔴"
                            status_text = "Available" if resource_status else "Unavailable"
                        else:
                            status_icon = "🟢"
                            status_text = str(resource_status).title()
                        display_name = resource_name.replace('_', ' ').title()
                        st.write(f"{status_icon} {display_name}: {status_text}")
                    
                    # Configuration Status
                    st.write("\n**⚙️ Configuration:**")
                    config = status.get("configuration", {})
                    st.write(f"🔵 Vision Model Type: {config.get('vision_model_type', 'Not Set').title()}")
                    st.write(f"🔵 Text Model Type: {config.get('text_model_type', 'Not Set').title()}")
                    st.write(f"🔵 Vision Model Name: {config.get('vision_model_name', 'Not Set')}")
                    st.write(f"🔵 Text Model Name: {config.get('text_model_name', 'Not Set')}")
                    st.write(f"🔵 Vision Temperature: {config.get('vision_temperature', 'Not Set')}")
                    st.write(f"🔵 Text Temperature: {config.get('text_temperature', 'Not Set')}")
                    
                    pixel_enabled = config.get('pixel_analysis_enabled', False)
                    st.write(f"{'🟢' if pixel_enabled else '🔴'} Pixel Analysis: {'Enabled' if pixel_enabled else 'Disabled'}")
                    
                    agents_enabled = config.get('individual_agents_enabled', False)
                    st.write(f"{'🟢' if agents_enabled else '🔴'} Individual Agents: {'Enabled' if agents_enabled else 'Disabled'}")
                
                # Advanced Details (Collapsible)
                st.markdown("---")
                st.markdown("### 🔍 Advanced Details")
                
                st.write("**System Timestamp:**", status.get('timestamp', 'Unknown'))
                
                if config.get('pixel_analysis_enabled'):
                    st.write(f"**Pixel Analysis Mode:** {'Fast' if config.get('fast_pixel_mode') else 'Detailed'}")
                    if config.get('pixel_block_size'):
                        st.write(f"**Pixel Block Size:** {config.get('pixel_block_size')}x{config.get('pixel_block_size')}")
                
                # Show any issues or warnings
                issues = []
                if not overall_health.get('apis_operational'):
                    issues.append("❌ No APIs are operational - analysis functionality limited")
                if not overall_health.get('dependencies_operational'):
                    issues.append("⚠️ Some dependencies are missing - functionality may be limited")
                if not overall_health.get('resources_operational'):
                    issues.append("⚠️ System resource issues detected")
                
                if issues:
                    st.write("**🚨 Issues Detected:**")
                    for issue in issues:
                        st.write(issue)
                else:
                    st.write("✅ **No issues detected** - System is fully operational")
                
            else:
                st.error("❌ Detector not initialized - System status unavailable")

    def run(self):
        """Main application entry point"""
        # Render sidebar configuration
        config = self.render_sidebar()
        
        # Render system status
        self.render_system_status()
        
        # Main content area
        uploaded_file = self.render_file_upload()
        
        if uploaded_file is not None:
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            # Column 1: Document Preview
            with col1:
                st.subheader("📄 Document Preview")
                self.render_document_preview(uploaded_file)
            
            # Column 2: Analysis Control and Results
            with col2:
                st.subheader("🔍 Analysis")
                
                # Display current configuration status
                with st.container():
                    st.write("**Current Configuration:**")
                    
                    # Show pixel forensics mode
                    pixel_mode = config["pixel_forensics_mode"]
                    mode_display = {
                        "disabled": "🚫 Disabled (Fastest)",
                        "basic_fast": "⚡ Basic Fast",
                        "enhanced_fast": "🔍 Enhanced Fast",
                        "enhanced_thorough": "🎯 Enhanced Thorough"
                    }.get(pixel_mode, pixel_mode)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Pixel Forensics:** {mode_display}")
                        if pixel_mode in ["enhanced_fast", "enhanced_thorough"]:
                            st.write(f"**Block Size:** {config['pixel_block_size']}x{config['pixel_block_size']}")
                    
                    with col_b:
                        st.write(f"**Individual Agents:** {'✅ Enabled' if config['enable_individual_agents'] else '🚫 Disabled'}")
                        st.write(f"**Visualization:** {'✅ Enabled' if config['enable_visualization'] else '🚫 Disabled'}")
                    
                    st.markdown("---")
                
                if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                    # Save uploaded file to temporary location
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_path = tmp_file.name
                    
                    try:
                        # Analyze the document
                        results = self.analyze_document(temp_file_path, config)
                        
                        if results:
                            # Display results in full width below
                            st.markdown("---")
                            self.render_analysis_results(results, config)
                    
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass

def main():
    """Main function to run the Streamlit app"""
    app = DocumentForgeryApp()
    app.run()

if __name__ == "__main__":
    main() 