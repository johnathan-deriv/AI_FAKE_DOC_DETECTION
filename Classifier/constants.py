import dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain.vectorstores import Chroma
from langchain.chains.query_constructor.base import get_query_constructor_prompt, StructuredQueryOutputParser,AttributeInfo
from datetime import datetime
# Root directory
root_dir = Path("Document Database/")
dotenv.load_dotenv()

# Updated environment variable names for Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro-preview-05-06")
gemini_embedding_model_name = os.getenv("GEMINI_EMBEDDING_MODEL_NAME", "models/text-embedding-004")

current_date=datetime.now().strftime("%d-%m-%Y")

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

# Using the same model for vision tasks
vision_model = model

# Initialize vector stores for different document types
document_vector_db=Chroma(embedding_function=embeddings, persist_directory="chroma_db_document_VS")
pot_vector_db=Chroma(embedding_function=embeddings, persist_directory="chroma_db_POT_VS")
real_documents_vector_db=Chroma(embedding_function=embeddings, persist_directory="chroma_db_real_documents_VS")


document_content_description="""
Information about the document type, issuing organization, and country of origin.
"""

metadata_field_info=[
    AttributeInfo(
        name="Document_type",
        description="the type of document (e.g., bank statement, medical bill, invoice, receipt)",
        type="string",
    ),
    AttributeInfo(
        name="Organization",
        description="the name of the organization that issued the document",
        type="string",
    ),
    AttributeInfo(
        name="Country",
        description="the country where the document was issued",
        type="string",
    ),
]

allowed_attributes = [
    "Document_type",
    "Organization",
    "Country",
]


# Define allowed comparators list
allowed_comparators = [
    "eq",  # Equal to (number, string, boolean)
    "ne",  # Not equal to (number, string, boolean)
    "exists", # Has the specified metadata field (boolean)
    "and", # Combines multiple filters
    "in", # In array (string or number)
    "nin", # Not in array (string or number)
]

examples = [
    (
        "Medical bill from Mayo Clinic in USA",
        {
            "query": "Medical bill from Mayo Clinic in USA",
            "filter": 'and(in("Document_type", ["medical bill"]),in("Organization", ["Mayo Clinic"]),in("Country", ["USA"]))',
        },
    ),
    (
        "Bank statement from a bank in Brazil",
        {
            "query": "Bank statement from a bank in Brazil",
            "filter": 'and(in("Document_type", ["bank statement"]),in("Country", ["Brazil"]))',
        },
    ),
]

# Create constructor prompt
constructor_prompt = get_query_constructor_prompt(
    document_content_description,
    metadata_field_info,
    allowed_comparators=allowed_comparators,
)
output_parser = StructuredQueryOutputParser.from_components(
    allowed_attributes=allowed_attributes,
    allowed_comparators=allowed_comparators,
)
query_constructor = constructor_prompt | model | output_parser
