from agent_RAG import rag_agent
from constants import model, real_documents_vector_db
from utils import Legitimacy_Info
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import base64
import io
from pdf2image import convert_from_path
from constants import current_date
import os

# Add function to handle path resolution
def resolve_document_path(file_path):
    """Resolves the path to a document file, handling both absolute and relative paths."""
    # Check if the path already contains 'Document Database'
    if 'Document Database' in file_path:
        # If it's an absolute path, return it directly
        if os.path.isabs(file_path):
            return file_path
        # If it's a relative path starting with 'Document Database'
        elif os.path.exists(file_path):
            return file_path
        # If the relative path exists with leading Document Database folder
        else:
            # Try with current directory
            base_dir = os.getcwd()
            absolute_path = os.path.join(base_dir, file_path)
            if os.path.exists(absolute_path):
                return absolute_path
            # Try one level up
            parent_dir = os.path.dirname(base_dir)
            absolute_path = os.path.join(parent_dir, file_path)
            if os.path.exists(absolute_path):
                return absolute_path
    else:
        # If path doesn't contain 'Document Database', try to find it
        base_dir = os.getcwd()
        doc_dir_path = os.path.join(base_dir, "Document Database")
        # If Document Database exists in current directory
        if os.path.exists(doc_dir_path):
            return os.path.join(doc_dir_path, file_path)
        # Try one level up
        parent_dir = os.path.dirname(base_dir)
        doc_dir_path = os.path.join(parent_dir, "Document Database")
        if os.path.exists(doc_dir_path):
            return os.path.join(doc_dir_path, file_path)
    
    # If all attempts fail, return the original path
    print(f"Warning: Could not resolve path for {file_path}")
    return file_path

def get_similar_real_documents(query, k=5):
    """
    Get similar real documents from the real_documents_vector_db.
    
    Args:
        query: The query to search for similar documents
        k: The number of similar documents to return
        
    Returns:
        List of similar documents with relevance scores
    """
    print(f"Searching for similar real documents with query: {query}")
    similar_documents = real_documents_vector_db.similarity_search_with_relevance_scores(query, k=k)
    similar_documents = sorted(similar_documents, key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(similar_documents)} similar real documents")
    for i, (doc, score) in enumerate(similar_documents):
        print(f"  Document {i+1} (Score: {score:.4f}): {doc.page_content[:100]}...")
    
    return similar_documents

def document_classifier_based_on_legit_docs(file, similar_legit_docs=None, number_of_pages=1):
    REAL_DOC = False
    print('COMPARING WITH SIMILAR LEGITIMATE DOCUMENTS')
    # Resolve the input file path
    file = resolve_document_path(file)
    
    # Early return if no legit documents to compare with
    if not similar_legit_docs:
        print("Warning: No legitimate documents found for comparison.")
        return True

    print(f"Entering RAG Comparison Analysis for {file}, comparing against {len(similar_legit_docs)} docs, using {number_of_pages} page(s).")

    # Case 1: Input file is an image
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        input_file_name = file.replace('../', '')
        input_image_path = str(input_file_name)
        try:
            input_image = Image.open(input_image_path)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_image_path}")
            return {"REAL_DOC": False, "LEGITIMACY_REASONING": f"Input file not found: {input_image_path}"}
        
        # Convert input image to base64
        buffer = io.BytesIO()
        input_image.save(buffer, format=input_image.format or "JPEG")
        base64_input_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # For each similar legit document
        for doc in similar_legit_docs:
            legit_file_name = doc[0].metadata['source_file'].replace('../', '')
            # Resolve the path to the legit file
            legit_file_name = resolve_document_path(legit_file_name)
            
            # Case 1A: Similar document is an image
            if legit_file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                legit_image_path = str(legit_file_name)
                legit_image = Image.open(legit_image_path)
                
                # Convert legit image to base64
                buffer = io.BytesIO()
                legit_image.save(buffer, format=legit_image.format or "JPEG")
                base64_legit_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Build message content for Image vs Image comparison
                response = model.with_structured_output(Legitimacy_Info).invoke(
                    [
                        SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. You are a helpful assistant that checks if two documents are similar. The current date is {current_date}."),
                        HumanMessage(content=[
                            {"type": "text", "text": f"""Extract the following information from the provided potentially fraudulent document image. The goal is to return whether the document is fake or not based on the following indicators:
                                1- Organization: Identify the organization that issued the document
                                2- Customer/recipient name
                                3- Digital artifacts: Check for mix of blurry parts and clear(sharp) parts, signs of digital manipulation 
                                4- Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles
                                5- Date verification: Check if dates in the document are reasonable and consistent
                                6- Layout issues: Check for misaligned elements, inconsistent formatting, or structural anomalies
                                7- Design verification: Evaluate if the document follows the expected design standards
                                8- Content analysis: Assess the reasonableness of content, amounts, dates, names, reference numbers, etc.
                            """
                            },
                            {"type": "text", "text": "Here is the image of the document to check:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/{input_image.format.lower() if input_image.format else 'jpeg'};base64,{base64_input_image}"}},
                            {"type": "text", "text": "Here is the similar document flagged as legitimate that you will be comparing to:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/{legit_image.format.lower() if legit_image.format else 'jpeg'};base64,{base64_legit_image}"}},
                            {"type": "text", "text": "Return the results containing the legitimacy: fake or legit and the reasoning for the result"},
                            {"type": "text", "text": 
                            f"""
                            Compare the document to the similar document labeled as legitimate based on the indicators listed previously.
                            Determine if the document is 'fake' or 'legit'.
                            
                            **Output Requirements:**
                            1.  Determine the overall `legitimacy` ('fake' or 'legit').
                            2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                            3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain **EACH** detected inconsistency type (e.g., Digital Artifacts, Layout Issues, Font Inconsistencies, etc.) with specific examples and text cited from the document for **EVERY** issue found. Do not just focus on one type of issue. If legit, return None for each indicator (example: Digital_artifacts: None, Font_inconsistencies: None, etc.).
                            """}
                        ])
                    ]
                )
            
            # Case 1B: Similar document is a PDF
            elif legit_file_name.endswith(".pdf"):
                # Convert PDF to images
                legit_pdf_path = str(legit_file_name)
                legit_pdf_images = convert_from_path(legit_pdf_path, dpi=300)
                legit_pages_to_process = legit_pdf_images[:number_of_pages]
                
                if not legit_pages_to_process:
                    print(f"  -> No pages extracted from PDF: {legit_pdf_path}")
                    continue  # Skip to next similar document
                
                # Build message content with input image and each page of the legit PDF
                message_content = [
                    {"type": "text", "text": f"""Extract the following information from the provided potentially fraudulent document image. The goal is to return whether the document is fake or not based on the following indicators:
                        1- Organization: Identify the organization that issued the document
                        2- Customer/recipient name
                        3- Digital artifacts: Check for mix of blurry parts and clear(sharp) parts, signs of digital manipulation 
                        4- Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles
                        5- Date verification: Check if dates in the document are reasonable and consistent
                        6- Layout issues: Check for misaligned elements, inconsistent formatting, or structural anomalies
                        7- Design verification: Evaluate if the document follows the expected design standards
                        8- Content analysis: Assess the reasonableness of content, amounts, dates, names, reference numbers, etc.
                    """
                    },
                    {"type": "text", "text": "Here is the image of the document to check:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/{input_image.format.lower() if input_image.format else 'jpeg'};base64,{base64_input_image}"}},
                    {"type": "text", "text": "Here is the similar document (PDF) labeled as legitimate that you will be comparing to:"},
                    {"type": "text", "text": 
                    f"""
                    Compare the document to the similar document (PDF) labeled as legitimate based on the indicators listed previously.
                    Determine if the document is 'fake' or 'legit'.
                    
                    **Output Requirements:**
                    1.  Determine the overall `legitimacy` ('fake' or 'legit').
                    2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                    3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain **EACH** detected inconsistency type (e.g., Digital Artifacts, Layout Issues, Font Inconsistencies, etc.) with specific examples and text cited from the document for **EVERY** issue found. Do not just focus on one type of issue. If legit, return None for each indicator (example: Digital_artifacts: None, Font_inconsistencies: None, etc.).
                    """}
                ]
                
                # Add each page of the legit PDF
                for i, legit_page in enumerate(legit_pages_to_process):
                    buffer = io.BytesIO()
                    legit_page.save(buffer, format="PNG")
                    base64_legit_page = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    message_content.append(
                        {"type": "text", "text": f"Page {i+1} of legit PDF:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_legit_page}"}}
                    )
                
                message_content.append(
                    {"type": "text", "text": "Return the results containing the legitimacy: fake or legit and the reasoning for the result"}
                )
                message_content.append(
                    {"type": "text", "text": 
                    f"""
                    Compare the document to the similar document labeled as legitimate based on the indicators listed previously.
                    Determine if the document is 'fake' or 'legit'.
                    
                    **Output Requirements:**
                    1.  Determine the overall `legitimacy` ('fake' or 'legit').
                    2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                    3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain **EACH** detected inconsistency type (e.g., Digital Artifacts, Layout Issues, Font Inconsistencies, etc.) with specific examples and text cited from the document for **EVERY** issue found. Do not just focus on one type of issue. If legit, return None for each indicator (example: Digital_artifacts: None, Font_inconsistencies: None, etc.).
                    """}
                )
                
                # Make the API call with combined content
                response = model.with_structured_output(Legitimacy_Info).invoke([
                    SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. You are a helpful assistant that checks if two documents are similar. The current date is {current_date}."),
                    HumanMessage(content=message_content)
                ])
            SIMILAR_DOCUMENT_NAME = legit_file_name
            SIMILAR_DOCUMENT_LEGITIMACY = "Legit"
            SIMILARITY_SCORE = doc[1]
            LEGITIMACY_REASONING = response.reasoning

            # set fake indicators to the indicators from response that are not None remove legitimacy and reasoning
            FAKE_INDICATORS = [key for key, value in response.__dict__.items() if value is not None and key not in ['legitimacy', 'reasoning']]
            # If we found it's legit, no need to check more
            if response.legitimacy == "legit":
                REAL_DOC = True
                break

    # Case 2: Input file is a PDF
    elif file.endswith(".pdf"):
        pdf_path = str(file)
        try:
            print(f"  RAG Comparison: Processing PDF ({number_of_pages} pages)...")
            pdf_images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=number_of_pages)
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {"REAL_DOC": False, "LEGITIMACY_REASONING": f"Error processing input PDF: {e}", "FAKE_INDICATORS": "ProcessingError"}
        
        print(f"Processing PDF with {len(pdf_images)} pages")
        pages_to_process = pdf_images[:number_of_pages]
        
        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            return False
        
        # For each similar legit document
        for doc in similar_legit_docs:
            legit_file_name = doc[0].metadata['source_file'].replace('../', '')
            # Resolve the path to the legit file
            legit_file_name = resolve_document_path(legit_file_name)
            
            # Case 2A: Similar document is an image
            if legit_file_name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
                # Load legit image
                legit_image_path = str(legit_file_name)
                legit_image = Image.open(legit_image_path)
                
                # Convert legit image to base64
                buffer = io.BytesIO()
                legit_image.save(buffer, format=legit_image.format or "JPEG")
                base64_legit_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Build message content starting with standard text
                message_content = [
                    {"type": "text", "text": f"""Extract the following information from the provided potentially fraudulent document PDF. The goal is to return whether the document is fake or not based on the following indicators:
                        1- Organization: Identify the organization that issued the document
                        2- Customer/recipient name
                        3- Digital artifacts: Check for mix of blurry parts and clear(sharp) parts, signs of digital manipulation 
                        4- Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles
                        5- Date verification: Check if dates in the document are reasonable and consistent
                        6- Layout issues: Check for misaligned elements, inconsistent formatting, or structural anomalies
                        7- Design verification: Evaluate if the document follows the expected design standards
                        8- Content analysis: Assess the reasonableness of content, amounts, dates, names, reference numbers, etc.
                    """
                    },
                    {"type": "text", "text": "Here are the pages of the PDF document to check:"}
                ]
                
                # Add each page of the input PDF
                for i, page in enumerate(pages_to_process):
                    buffer = io.BytesIO()
                    page.save(buffer, format="PNG")
                    base64_page = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    message_content.append(
                        {"type": "text", "text": f"Page {i+1} of input PDF:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_page}"}}
                    )
                    print(f"Page {i+1} of input PDF added to message content")
                
                # Add the legit image
                message_content.append(
                    {"type": "text", "text": "Here is the similar document flagged as legitimate that you will be comparing to:"}
                )
                message_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/{legit_image.format.lower() if legit_image.format else 'jpeg'};base64,{base64_legit_image}"}}
                )
                message_content.append(
                        {"type": "text", "text": "Return the results containing the legitimacy: fake or legit and the reasoning for the result"}
                )
                message_content.append(
                    {"type": "text", "text": 
                    f"""
                    Compare the document to the similar document labeled as legitimate based on the indicators listed previously.
                    Determine if the document is 'fake' or 'legit'.
                    
                    **Output Requirements:**
                    1.  Determine the overall `legitimacy` ('fake' or 'legit').
                    2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                    3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain **EACH** detected inconsistency type (e.g., Digital Artifacts, Layout Issues, Font Inconsistencies, etc.) with specific examples and text cited from the document for **EVERY** issue found. Do not just focus on one type of issue. If legit, return None for each indicator (example: Digital_artifacts: None, Font_inconsistencies: None, etc.).
                    """}
                )
                # Make the API call with combined content
                response = model.with_structured_output(Legitimacy_Info).invoke([
                    SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. You are a helpful assistant that checks if two documents are similar. The current date is {current_date}."),
                    HumanMessage(content=message_content)
                ])
                
            # Case 2B: Similar document is a PDF
            elif legit_file_name.endswith(".pdf"):
                # Convert legit PDF to images
                legit_pdf_path = str(legit_file_name)
                legit_pdf_images = convert_from_path(legit_pdf_path, dpi=300)
                legit_pages_to_process = legit_pdf_images[:number_of_pages]
                
                if not legit_pages_to_process:
                    print(f"  -> No pages extracted from PDF: {legit_pdf_path}")
                    continue  # Skip to next similar document
                
                # Build message content starting with standard text
                message_content = [
                    {"type": "text", "text": f"""Extract the following information from the provided potentially fraudulent document PDF. The goal is to return whether the document is fake or not based on the following indicators:
                        1- Organization: Identify the organization that issued the document
                        2- Customer/recipient name
                        3- Digital artifacts: Check for mix of blurry parts and clear(sharp) parts, signs of digital manipulation 
                        4- Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles
                        5- Date verification: Check if dates in the document are reasonable and consistent
                        6- Layout issues: Check for misaligned elements, inconsistent formatting, or structural anomalies
                        7- Design verification: Evaluate if the document follows the expected design standards
                        8- Content analysis: Assess the reasonableness of content, amounts, dates, names, reference numbers, etc.
                    """
                    },
                    {"type": "text", "text": "Here are the pages of the input PDF document to check:"}
                ]
                
                # Add each page of the input PDF
                for i, page in enumerate(pages_to_process):
                    buffer = io.BytesIO()
                    page.save(buffer, format="PNG")
                    base64_page = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    message_content.append(
                        {"type": "text", "text": f"Page {i+1} of input PDF:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_page}"}}
                    )
                    print(f"Page {i+1} of input PDF added to message content")
                # Add text explaining the legit PDF
                message_content.append(
                    {"type": "text", "text": "Here are the pages of the similar document (PDF) flagged as legitimate that you will be comparing to:"}
                )
                
                # Add each page of the legit PDF
                for i, legit_page in enumerate(legit_pages_to_process):
                    buffer = io.BytesIO()
                    legit_page.save(buffer, format="PNG")
                    base64_legit_page = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    message_content.append(
                        {"type": "text", "text": f"Page {i+1} of legit PDF:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_legit_page}"}}
                    )
                
                message_content.append(
                    {"type": "text", "text": "Return the results containing the legitimacy: fake or legit and the reasoning for the result"}
                )

                message_content.append(
                    {"type": "text", "text": 
                    f"""
                    Compare the document to the similar document labeled as legitimate based on the indicators listed previously.
                    Determine if the document is 'fake' or 'legit'.
                    
                    **Output Requirements:**
                    1.  Determine the overall `legitimacy` ('fake' or 'legit').
                    2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                    3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain **EACH** detected inconsistency type (e.g., Digital Artifacts, Layout Issues, Font Inconsistencies, etc.) with specific examples and text cited from the document for **EVERY** issue found. Do not just focus on one type of issue. If legit, return None for each indicator (example: Digital_artifacts: None, Font_inconsistencies: None, etc.).
                    """}
                )
                # Make the API call with combined content
                response = model.with_structured_output(Legitimacy_Info).invoke([
                    SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. You are a helpful assistant that checks if two documents are similar. The current date is {current_date}."),
                    HumanMessage(content=message_content)
                ])

            SIMILAR_DOCUMENT_NAME = legit_file_name
            SIMILAR_DOCUMENT_LEGITIMACY = "Legit"
            SIMILARITY_SCORE = doc[1]
            LEGITIMACY_REASONING = response.reasoning

            # set fake indicators to the indicators from response that are not None remove legitimacy and reasoning
            FAKE_INDICATORS = [key for key, value in response.__dict__.items() if value is not None and key not in ['legitimacy', 'reasoning']]
            # If we found it's legit, no need to check more
            if response.legitimacy == "legit":
                REAL_DOC = True
                break

    # remove duplicates from FAKE_INDICATORS
    FAKE_INDICATORS = list(set(FAKE_INDICATORS))
    # to string
    FAKE_INDICATORS = ", ".join(FAKE_INDICATORS)

    # return the results as dictionary
    return {"REAL_DOC": REAL_DOC,
            "SIMILAR_DOCUMENT_NAME": SIMILAR_DOCUMENT_NAME,
            "SIMILAR_DOCUMENT_LEGITIMACY": SIMILAR_DOCUMENT_LEGITIMACY,
            "SIMILARITY_SCORE": SIMILARITY_SCORE,
            "LEGITIMACY_REASONING": LEGITIMACY_REASONING,
            "FAKE_INDICATORS": FAKE_INDICATORS}

def document_classifier_with_real_documents(file, number_of_pages=1, k=5):
    """
    Classify a document as real or fake by comparing it with real documents from the vector store.
    
    Args:
        file: Path to the document file to classify
        number_of_pages: Number of pages to process for PDF documents
        k: Number of similar real documents to retrieve from the vector store
        
    Returns:
        Dictionary with classification results
    """
    from agent_OCR import ocr_agent
    
    print('CLASSIFYING DOCUMENT USING REAL DOCUMENTS VECTOR STORE')
    
    # Resolve the input file path
    file = resolve_document_path(file)
    
    # Extract information from the document using OCR
    ocr_info = ocr_agent(file, number_of_pages)
    
    # Build query from OCR information
    query_parts = []
    if ocr_info.Document_type:
        query_parts.append(f"Document type: {ocr_info.Document_type}")
    if ocr_info.Organization:
        query_parts.append(f"Organization: {ocr_info.Organization}")
    if ocr_info.Country:
        query_parts.append(f"Country: {ocr_info.Country}")
    if ocr_info.Customer_name:
        query_parts.append(f"Customer: {ocr_info.Customer_name}")
    
    # If no information was extracted, return early
    if len(query_parts) == 0:
        print("Warning: No information extracted from document. Cannot perform comparison.")
        return {"REAL_DOC": False, 
                "LEGITIMACY_REASONING": "Could not extract information from document for comparison", 
                "FAKE_INDICATORS": "Information Extraction Failure"}
    
    # Build query string
    query = ". ".join(query_parts)
    print(f"Query for similar real documents: {query}")
    
    # Get similar real documents
    similar_real_docs = get_similar_real_documents(query, k=k)
    
    # If no similar documents found, return early
    if not similar_real_docs:
        print("Warning: No similar real documents found for comparison.")
        return {"REAL_DOC": False, 
                "LEGITIMACY_REASONING": "No similar real documents found for comparison", 
                "FAKE_INDICATORS": "No Reference Documents"}
    
    # Use the existing function to compare with similar real documents
    result = document_classifier_based_on_legit_docs(file, similar_real_docs, number_of_pages)
    
    return result
