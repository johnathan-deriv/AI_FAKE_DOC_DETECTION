from agent_OCR import ocr_agent
from constants import document_vector_db, pot_vector_db


def rag_agent(file, number_of_pages=1):
    ocr_info = ocr_agent(file, number_of_pages)
    document_type = ocr_info.Document_type
    organization = ocr_info.Organization
    country = ocr_info.Country
    customer_name = ocr_info.Customer_name

    query_parts = []
    if document_type:
        query_parts.append(f"Document type: {document_type}")
    if organization:
        query_parts.append(f"Organization: {organization}")
    if country:
        query_parts.append(f"Country: {country}")
    if customer_name:
        query_parts.append(f"Customer: {customer_name}")
    
    if len(query_parts) == 0:
        similar_documents = []
    else:
        query = ". ".join(query_parts)
        
        # Choose the appropriate vector store based on document type
        if document_type and "transaction" in document_type.lower():
            # For POT (Proof of Transaction) documents
            vector_db = pot_vector_db
            print(f"Using POT vector store for document type: {document_type}")
        else:
            # For all other document types
            vector_db = document_vector_db
            print(f"Using general document vector store for document type: {document_type}")
        
        similar_documents = vector_db.similarity_search_with_relevance_scores(query)
        similar_documents = sorted(similar_documents, key=lambda x: x[1], reverse=True)
    
    return similar_documents
