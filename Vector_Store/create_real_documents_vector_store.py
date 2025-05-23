import os
import sys
from pathlib import Path
import dotenv
from PIL import Image
import io
import base64
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage

# Add the parent directory to the path so we can import from Classifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Classifier.utils import OCR_Info

# Load environment variables
dotenv.load_dotenv()

# Get API keys and model names
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-05-06")
gemini_embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/text-embedding-004")

# Initialize embeddings and model
embeddings = GoogleGenerativeAIEmbeddings(
    model=gemini_embedding_model_name,
    google_api_key=google_api_key
)

model = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    google_api_key=google_api_key,
    temperature=0,
    convert_system_message_to_human=True
)

def extract_document_info(file_path):
    """
    Extract document information from an image file using the vision model.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        OCR_Info object containing document information
    """
    print(f"Processing {file_path}")
    
    # Ensure file_path is a string for subsequent operations that rely on string methods
    # (e.g., str.endswith). This allows the function to accept either str or pathlib.Path
    # objects without raising attribute errors.
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    if file_path.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file_path)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None
            
        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # Extract document information using the model
        try:
            response = model.with_structured_output(OCR_Info).invoke(
                [
                    SystemMessage(
                        content="""
                        You are analyzing official documents to extract key information.
                        Extract only the information that is clearly visible in the document.
                        """
                    ),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": 
                                f"""Extract the following information from the provided document image:
                                - Document_type: Identify the type of document (e.g., contract, agreement, form, certificate, etc.)
                                - Organization: Identify the organization that issued the document
                                - Country: Identify the country where the document was issued
                                - Customer_name: Identify the name of the customer/recipient if present
                                
                                Only include information that is explicitly visible in the document.
                                """
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}}
                        ]
                    )
                ]
            )
            return response
        except Exception as e:
            print(f"Error extracting information from {image_path}: {e}")
            return None
    else:
        print(f"Unsupported file format: {file_path}")
        return None

def process_file_to_document(file_path):
    """
    Process a file into a Langchain Document with metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Document object if successful, None otherwise
    """
    print(f"Processing {file_path} into Document...")
    
    try:
        # Extract document information
        info_result = extract_document_info(file_path)
        
        if info_result is None:
            print(f"  -> Warning: extract_document_info returned None for {file_path}. Skipping Document creation.")
            return None
        
        # Construct page content
        content_parts = []
        
        document_type = getattr(info_result, 'Document_type', "Unknown Document Type") or "Unknown Document Type"
        content_parts.append(f"Document Type: {document_type}")
        
        organization = getattr(info_result, 'Organization', "Unknown Organization") or "Unknown Organization"
        content_parts.append(f"Organization: {organization}")
        
        country = getattr(info_result, 'Country', "Unknown Country") or "Unknown Country"
        content_parts.append(f"Country: {country}")
        
        customer_name = getattr(info_result, 'Customer_name', None)
        if customer_name:
            content_parts.append(f"Customer: {customer_name}")
        
        page_content_str = ". ".join(content_parts)
        print(f"  Generated page_content: {page_content_str[:200]}...")
        
        # Construct metadata
        metadata_dict = {
            'Document_type': document_type,
            'Organization': organization,
            'Country': country,
            'Customer_name': customer_name if customer_name else "Unknown Customer",
            'source_file': str(file_path)
        }
        
        # Create Document
        doc = Document(page_content=page_content_str, metadata=metadata_dict)
        print(f"  -> Successfully created Document for {file_path}")
        return doc
    
    except Exception as e:
        print(f"  -> Error processing {file_path} into Document: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Path to real documents
    real_documents_dir = Path("Vector_Store/real_documents")
    
    # Check if directory exists
    if not real_documents_dir.exists():
        print(f"Directory {real_documents_dir} does not exist.")
        return
    
    # Process documents
    document_list = []
    processed_count = 0
    skipped_count = 0
    
    print(f"Starting processing of real documents in: {real_documents_dir.resolve()}")
    
    # Iterate through files in the directory
    for file_path in real_documents_dir.iterdir():
        # Skip directories and hidden files
        if file_path.is_file() and not file_path.name.startswith('.'):
            # Check for supported extensions
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
                doc = process_file_to_document(file_path)
                if doc:
                    document_list.append(doc)
                    processed_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"    Skipping unsupported file type: {file_path.name}")
                skipped_count += 1
    
    print(f"\n--- Document Processing Complete ---")
    print(f"Successfully processed {processed_count} real documents.")
    print(f"Skipped {skipped_count} files (errors or unsupported types).")
    
    # Create vector store if documents were processed
    if document_list:
        print("\nCreating vector store...")
        vector_db = Chroma.from_documents(
            document_list, 
            embeddings, 
            persist_directory="Classifier/chroma_db_real_documents_VS"
        )
        print(f"Vector store created and saved to: Classifier/chroma_db_real_documents_VS")
        
        # Test the vector store
        print("\nTesting vector store with a sample query...")
        results = vector_db.similarity_search_with_relevance_scores("official document", k=2)
        print(f"Found {len(results)} similar documents")
        for i, (doc, score) in enumerate(results):
            print(f"\nResult {i+1} (Score: {score:.4f}):")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
    else:
        print("No documents were processed. Vector store not created.")

if __name__ == "__main__":
    main()
