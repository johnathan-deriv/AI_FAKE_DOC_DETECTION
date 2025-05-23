from utils import OCR_Info
from constants import model
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import base64
import io
from pdf2image import convert_from_path
import json

def ocr_agent(file, number_of_pages=1):
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = model.with_structured_output(OCR_Info).invoke(
            [
                SystemMessage(
                content="""
                You are analyzing various types of documents to detect potential forgeries. These could include financial documents, 
                medical bills, invoices, receipts, or any other official documents.
                """
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": 
                            f"""Extract the following information from the provided document image. Only include details explicitly visible in the document. Do not add fields or values if they are not present:
                            - Document_type : identify the type of document (e.g., bank statement, medical bill, invoice, receipt)
                            - Organization : identify the name of the organization that issued the document
                            - Country : identify the country where the document was issued (can be guessed from the organization name, language, or directly from the document)
                            - Customer_name : name of the customer/patient/recipient
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}}
                ]
            ),
            ]
        )

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=600)
        print(f"Processing PDF with {len(pdf_images)} pages")
        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result

        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Extract the following information from the provided PDF document pages. Only include details explicitly visible in the document. Do not add fields or values if they are not present:
                - Document_type : identify the type of document (e.g., bank statement, medical bill, invoice, receipt)
                - Organization : identify the name of the organization that issued the document
                - Country : identify the country where the document was issued
                - Customer_name : name of the customer/patient/recipient
                """
            }
        )

        # Loop through pages, convert to base64, and add image blocks
        for i, image in enumerate(pages_to_process):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            # Append each image as a separate dictionary in the content list
            message_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            )
            print(f"  -> Added page {i+1} to the request")

        # Make ONE API call with all text and image content
        response = model.with_structured_output(OCR_Info).invoke(
            [
                SystemMessage(
                    content="""
                    You are analyzing various types of documents to detect potential forgeries. These could include financial documents, 
                    medical bills, invoices, receipts, or any other official documents.
                    """
                ),
                HumanMessage(content=message_content), # Pass the list with text and multiple images
            ]
        )
        

    return response
