# Document Forgery Detection System

🔍 **Advanced AI-powered document forgery detection system with no-training approach**

## 🚀 Recent Migration Update

This system has been enhanced with advanced components migrated from a previous classifier system, including:

### ✨ New Features Added
- **Enhanced OCR with EasyOCR**: Superior text detection with rotated bounding box support
- **Individual Analysis Agents**: 6 specialized agents for different forgery types
- **Advanced Visualization**: Sophisticated bounding box overlays with color coding
- **Modern Web Interface**: Streamlit app with real-time analysis and visualization
- **Multiple API Support**: Both OpenAI and Google Gemini integration
- **Comprehensive Data Models**: Pydantic models for structured outputs

### 🔧 Enhanced Components

#### 1. Enhanced OCR (`src/enhanced_ocr.py`)
- **EasyOCR + Tesseract Integration**: Best-of-both-worlds approach
- **Rotated Bounding Boxes**: Handles angled text detection
- **Fuzzy Text Matching**: Find similar text with fuzzywuzzy
- **Document Type Inference**: Automatic document classification
- **IoU Overlap Detection**: Prevents duplicate bounding boxes

#### 2. Individual Analysis Agents (`src/individual_agents.py`)
- **Digital Artifacts Agent**: Detects digital tampering signs
- **Font Inconsistencies Agent**: Identifies mismatched fonts
- **Date Verification Agent**: Checks for impossible dates
- **Layout Issues Agent**: Finds alignment problems
- **Design Verification Agent**: Validates logos and branding
- **Content Analysis Agent**: Logical consistency checking

#### 3. Advanced Visualization (`src/visualization.py`)
- **Color-coded Indicators**: 7 different colors for indicator types
- **Multi-page Support**: Handles PDF documents
- **Smart Overlap Detection**: Prevents cluttered annotations
- **HTML Legends**: Interactive legend generation
- **Interactive Hover Details**: Hover over annotated regions to see detailed analysis information
- **Plotly Integration**: Modern interactive visualizations with zoom and pan capabilities
- **Fallback Overlays**: Simple borders when specific regions unavailable

#### 4. Modern Web Application (`src/web_app.py`)
- **Real-time Analysis**: Upload and analyze instantly
- **Interactive Configuration**: Sidebar controls for all settings
- **Module Results Display**: Tabbed interface for detailed results
- **System Status Monitoring**: API and module health checks
- **Beautiful UI**: Modern styling with custom CSS

## 📋 System Architecture

```
src/
├── models.py              # Pydantic data models
├── enhanced_ocr.py        # Advanced OCR with EasyOCR + Tesseract
├── individual_agents.py   # Specialized forgery detection agents
├── visualization.py       # Advanced visualization with bounding boxes
├── web_app.py            # Modern Streamlit web interface
├── detector.py           # Main document forgery detector
├── ingest.py             # Document ingestion and preprocessing
├── vision_net.py         # Computer vision analysis
├── layout_reasoner.py    # Layout and structure analysis
├── fusion.py             # Multi-modal analysis fusion
├── pdf_meta.py           # PDF metadata analysis
└── report.py             # Comprehensive reporting
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (optional)
- Google Gemini API key (optional)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd AI_FAKE_DOC_DETECTION
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (create `.env` file):
```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Model Names (optional - defaults provided)
OPENAI_MODEL_NAME=gpt-4o
GEMINI_MODEL_NAME=gemini-2.5-pro-preview-05-06

# Optional Settings
DEBUG=false
LOG_LEVEL=INFO
```

## 🚀 Usage

### Web Application (Recommended)
Launch the modern web interface:
```bash
python run_web_app.py
```
Then open http://localhost:8501 in your browser.

### Direct Usage
```python
from src.detector import DocumentForgeryDetector

# Initialize detector
detector = DocumentForgeryDetector()

# Analyze document
results = detector.analyze_document("path/to/document.pdf")

# Check results
if results["verdict"] == "suspicious":
    print("⚠️ Document may be forged!")
    print(f"Confidence: {results['overall_score']:.3f}")
```

## 🎯 Features

### Core Capabilities
- **Multi-format Support**: PDF, JPG, PNG, TIFF, WebP
- **No Training Required**: Pre-trained models and rule-based analysis
- **Real-time Analysis**: Fast processing with caching
- **Comprehensive Reports**: Detailed reasoning and evidence
- **Visual Annotations**: Highlight suspicious regions
- **Interactive Visualization**: Hover over detected regions to see detailed analysis information, confidence scores, and source module data

### Analysis Modules
1. **Vision Analysis**: Computer vision-based tampering detection
2. **Layout Reasoning**: Document structure and formatting analysis
3. **PDF Metadata**: Hidden metadata inconsistencies
4. **OCR Enhancement**: Text extraction and verification
5. **Individual Agents**: Specialized forgery type detection
6. **Fusion Analysis**: Multi-modal result combination

### Supported Document Types
- Financial documents (bank statements, invoices)
- Medical bills and records
- Insurance documents
- Tax forms and certificates
- Receipts and vouchers
- Contracts and agreements

## 📊 Output Format

The system provides structured output with:

```python
{
    "verdict": "suspicious|legitimate",
    "overall_score": 0.85,
    "reasoning": "Detailed explanation...",
    "indicators": ["digital_artifacts", "font_inconsistencies"],
    "module_results": {
        "vision_analysis": {...},
        "individual_agents": {...},
        # ... other modules
    },
    "visualization": {
        "annotated_images": [...],
        "legend_html": "...",
    }
}
```

## 🔧 Configuration

### Web App Settings
- **API Selection**: Choose between OpenAI and Gemini
- **Analysis Options**: Enable/disable individual modules
- **Confidence Thresholds**: Adjust sensitivity
- **PDF Processing**: Set max pages to analyze
- **Visualization Mode**: Toggle between interactive (with hover details) and static visualization

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=sk-...                        # OpenAI API key
GEMINI_API_KEY=AIza...                       # Google Gemini API key (preferred)
GOOGLE_API_KEY=AIza...                       # Alternative name for Gemini API key

# Model Names (optional)
OPENAI_MODEL_NAME=gpt-4o                     # OpenAI model to use
GEMINI_MODEL_NAME=gemini-2.5-pro-preview-05-06  # Gemini model to use

# Optional Settings
DEBUG=false                                  # Enable debug mode
LOG_LEVEL=INFO                              # Logging level
MAX_FILE_SIZE=50                            # Max file size in MB
```

## 📈 Performance

- **Speed**: ~5-15 seconds per document
- **Accuracy**: High precision with multiple validation layers
- **Memory**: Optimized for standard desktop systems
- **Scalability**: Async processing for batch operations

## 🔍 Technical Details

### Enhanced OCR Pipeline
1. **EasyOCR**: Advanced text detection with bounding boxes
2. **Tesseract Fallback**: Backup OCR for reliability
3. **Text Matching**: Fuzzy matching for indicator correlation
4. **Coordinate Normalization**: Consistent bounding box formats

### Individual Agent Architecture
- **Specialized Prompts**: Tailored for specific forgery types
- **Multi-API Support**: OpenAI GPT-4o and Google Gemini
- **PDF Processing**: Multi-page analysis with pdf2image
- **Error Handling**: Graceful fallbacks and error recovery

### Visualization Engine
- **OpenCV Integration**: Advanced image processing
- **Color Mapping**: Consistent color coding for indicator types
- **Overlap Detection**: IoU-based duplicate prevention
- **Multi-format Output**: Support for various image formats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:
1. Check the documentation
2. Create an issue on GitHub
3. Contact the development team

---

**Built with ❤️ for document security and fraud prevention**
