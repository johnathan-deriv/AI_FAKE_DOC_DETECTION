# AI Document Forgery Detector

This system is designed to detect fake or forged documents of various types, including financial documents, medical bills, invoices, receipts, and more. It uses a combination of AI techniques to analyze documents and determine their legitimacy.

## System Overview

The document forgery detection system works in two main ways:

1. **Comparison with Known Legitimate Documents**: The system compares the input document with a database of known legitimate documents to identify inconsistencies.
2. **Individual Forgery Checks**: If no similar legitimate documents are found, the system runs a series of specialized checks to identify common forgery indicators.

## Key Components

- **Document Classifier**: The main entry point that orchestrates the document analysis process.
- **RAG Agent**: Retrieves similar legitimate documents from the vector store for comparison.
- **Individual Check Agents**: Specialized agents that check for specific forgery indicators:
  - Digital Artifacts Check
  - Font Inconsistencies Check
  - Layout Issues Check
  - Design Verification Check
  - Content Analysis Check
  - Date Verification Check
- **Vector Stores**: Databases of document embeddings for similarity search:
  - General Document Vector Store: For various document types
  - POT Vector Store: Specifically for Proof of Transaction documents

## Setup and Usage

### Prerequisites

1. Python 3.8+
2. Required packages (install using `pip install -r Classifier/requirements.txt`)
3. Google API key for Gemini model access

### Environment Setup

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-pro-preview-05-06
GEMINI_EMBEDDING_MODEL_NAME=models/embedding-001
```

### Creating Document Embeddings

1. Place legitimate document images in the `Vector Store/real_documents/` directory.
2. Run the embedding creation notebooks:
   ```
   jupyter notebook "Vector Store/Create_document_embeddings.ipynb"
   jupyter notebook "Vector Store/Create_vector_store.ipynb"
   ```
3. This will create a vector store in the `Classifier/chroma_db_document_VS/` directory.

### Running the Detector

You can use the system through the web interface or API:

1. **Web Interface**:
   ```
   cd Classifier
   streamlit run app.py
   ```
   Then open your browser to http://localhost:8501

2. **API**:
   ```
   cd Classifier
   python api.py
   ```
   The API will be available at http://localhost:5000

## How It Works

1. The system extracts key information from the document using OCR.
2. It searches for similar legitimate documents in the vector store.
3. If similar documents are found, it compares the input document with them.
4. If no similar documents are found or if comparison suggests forgery, it runs individual checks.
5. Based on the results, it determines if the document is fake or legitimate.
6. For fake documents, it visualizes the forgery indicators.

## Vector Stores

The system uses two vector stores:

1. **General Document Vector Store** (`chroma_db_document_VS`): Contains embeddings for various document types.
2. **POT Vector Store** (`chroma_db_POT_VS`): Contains embeddings specifically for Proof of Transaction documents.

The system automatically selects the appropriate vector store based on the document type.

## Extending the System

To add support for new document types:

1. Add legitimate examples of the new document type to `Vector Store/real_documents/`.
2. Run the embedding creation notebooks to update the vector store.
3. The system will automatically use these new documents for comparison.

## Visualization

When a document is identified as fake, the system attempts to visualize the forgery indicators by:

1. Highlighting specific text elements associated with forgery indicators.
2. Color-coding different types of forgery indicators.
3. If specific text elements can't be identified, adding a colored border around the document.
