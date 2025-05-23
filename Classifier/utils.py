from pydantic import BaseModel, Field
from typing import Optional
from typing import Any, Dict
from typing import List
from langchain_core.documents import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever


class forgery_text(BaseModel):
    Digital_artifacts: Optional[str] = Field(default=None, description="the text from the image that corresponds to the digital artifacts forgery")
    Font_inconsistencies: Optional[str] = Field(default=None, description="the text from the image that corresponds to the font inconsistencies forgery")
    Date_verification: Optional[str] = Field(default=None, description="the text from the image that corresponds to the date verification forgery")
    Layout_issues: Optional[str] = Field(default=None, description="the text from the image that corresponds to the layout issues forgery")
    Design_verification: Optional[str] = Field(default=None, description="the text from the image that corresponds to the design verification forgery")
    Content_analysis: Optional[str] = Field(default=None, description="the text from the image that corresponds to the content analysis forgery")
    

class OCR_Info(BaseModel):
    """
    A class representing the information extracted from a document.
    """
    Document_type: Optional[str] = Field(default=None, description="the type of document (e.g., bank statement, medical bill, invoice, receipt)")
    Organization: Optional[str] = Field(default=None, description="the name of the organization that issued the document")
    Country: Optional[str] = Field(default=None, description="the country where the document was issued")
    Customer_name: Optional[str] = Field(default=None, description="the name of the customer/patient/recipient")
    
# Custom Self Query Retriever to add score information to the metadata retriever
class CustomSelfQueryRetriever(SelfQueryRetriever):
    def _get_docs_with_query(
        self, query: str, search_kwargs: Dict[str, Any]
    ) -> List[Document]:
        """Get docs, adding score information."""
        try:
            docs, scores = zip(
                *self.vectorstore.similarity_search_with_relevance_scores(query, **search_kwargs)
            )
            new_docs=[]
            for doc, score in zip(docs, scores):
                doc = (doc,score)
                new_docs.append(doc)
            docs=new_docs
        except Exception as e:
            return []

        return docs
    
class Legitimacy_Info(BaseModel):
    Digital_artifacts: Optional[str] = Field(default=None, description="the text from the image that corresponds to the digital artifacts forgery, if not found return None")
    Font_inconsistencies: Optional[str] = Field(default=None, description="the text from the image that corresponds to the font inconsistencies forgery, if not found return None")
    Date_verification: Optional[str] = Field(default=None, description="the text from the image that corresponds to the date verification forgery, if not found return None")
    Layout_issues: Optional[str] = Field(default=None, description="the text from the image that corresponds to the layout issues forgery, if not found return None")
    Design_verification: Optional[str] = Field(default=None, description="the text from the image that corresponds to the design verification forgery, if not found return None")
    Content_analysis: Optional[str] = Field(default=None, description="the text from the image that corresponds to the content analysis forgery, if not found return None")
    legitimacy: Optional[str] = Field(default=None, description="the legitimacy of the document: fake or legit")
    reasoning: Optional[str] = Field(default=None, description="the reasoning for the legitimacy of the document")


class Individual_legitimacy_Info(BaseModel):
    legitimacy: Optional[str] = Field(default=None, description="the legitimacy of the document: fake or legit")
    reasoning: Optional[str] = Field(default=None, description="the reasoning for the legitimacy of the document")


class Final_analysis_Info(BaseModel):
    legitimacy: Optional[str] = Field(default=None, description="the legitimacy of the document: fake or legit")
    reasoning: Optional[str] = Field(default=None, description="the reasoning for the legitimacy of the document")    
    forgery_indicators: Optional[List[str]] = Field(default=None, description="the forgery indicators of the document")
