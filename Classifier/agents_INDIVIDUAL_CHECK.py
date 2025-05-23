
from constants import model
from utils import Individual_legitimacy_Info,Final_analysis_Info
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import base64
import io
from pdf2image import convert_from_path
from constants import current_date


def Digital_artifacts_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Digital artifacts: Check for mix of blurry/sharp parts, manipulation signs.
                                               Forged documents may have low-quality images, distorted logos, or altered fonts that differ from official records.
                                               Look for official logos or watermarks on documents. Forged documents may lack these, or the design might be inconsistent with the official logo.
                                               Pay attention to pixelation around text, uneven compression artifacts, or signs of digital tampering.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Digital Artifacts, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Digital artifacts: Check for mix of blurry/sharp parts, manipulation signs.
                                               Forged documents may have low-quality images, distorted logos, or altered fonts that differ from official records.
                                               Look for official logos or watermarks on documents. Forged documents may lack these, or the design might be inconsistent with the official logo.
                                               Pay attention to pixelation around text, uneven compression artifacts, or signs of digital tampering.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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


        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Digital Artifacts, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}

def Font_inconsistencies_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles in illogical places.
                                                 A legitimate document typically maintains consistent font styling throughout,
                                                 while forgeries often show variations where information has been altered.
                                                 Pay special attention to important fields like names, dates, amounts, and reference numbers.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Font Inconsistencies, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Font inconsistencies: Check for mismatched fonts, sizes, weights, or styles in illogical places.
                                                 A legitimate document typically maintains consistent font styling throughout,
                                                 while forgeries often show variations where information has been altered.
                                                 Pay special attention to important fields like names, dates, amounts, and reference numbers.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Font Inconsistencies, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}

def Date_verification_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Date verification: Examine all dates in the document for logical consistency. 
                                              Check if dates are in the future (relative to the current date {current_date}),
                                              if timestamps make sense (not at impossible hours),
                                              and if multiple dates within the document are consistent with each other.
                                              For medical documents, check if treatment dates align with visit dates.
                                              For financial documents, verify if transaction dates match statement periods.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Date Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Date verification: Examine all dates in the document for logical consistency. 
                                              Check if dates are in the future (relative to the current date {current_date}),
                                              if timestamps make sense (not at impossible hours),
                                              and if multiple dates within the document are consistent with each other.
                                              For medical documents, check if treatment dates align with visit dates.
                                              For financial documents, verify if transaction dates match statement periods.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Date Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}

def Layout_issues_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Layout issues: Examine the overall document structure and formatting.
                                          Check for misaligned text/elements, inconsistent spacing, unprofessional design,
                                          overlapping elements, or layout differences from standard templates for this type of document.
                                          Legitimate documents typically have consistent, professional layouts with proper alignment
                                          and spacing between sections.
                                          For medical documents, check for proper header/footer placement and consistent section formatting.
                                          For financial documents, verify proper column alignment and consistent table formatting.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Layout Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Layout issues: Examine the overall document structure and formatting.
                                          Check for misaligned text/elements, inconsistent spacing, unprofessional design,
                                          overlapping elements, or layout differences from standard templates for this type of document.
                                          Legitimate documents typically have consistent, professional layouts with proper alignment
                                          and spacing between sections.
                                          For medical documents, check for proper header/footer placement and consistent section formatting.
                                          For financial documents, verify proper column alignment and consistent table formatting.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Layout Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}

def Design_verification_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Design verification: Evaluate if the document follows the expected design standards of the claimed issuing organization.
                                               Check for proper branding elements (logos, colors, headers, footers),
                                               security features (watermarks, digital stamps, QR codes), and overall 
                                               formatting consistent with that organization's typical documents.
                                               For financial documents, verify bank logos, color schemes, and security features.
                                               For medical documents, check for hospital/clinic letterhead, proper medical symbols, and standard formatting.
                                               For invoices/receipts, verify company branding, standard layout, and expected design elements.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Design Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Design verification: Evaluate if the document follows the expected design standards of the claimed issuing organization.
                                               Check for proper branding elements (logos, colors, headers, footers),
                                               security features (watermarks, digital stamps, QR codes), and overall 
                                               formatting consistent with that organization's typical documents.
                                               For financial documents, verify bank logos, color schemes, and security features.
                                               For medical documents, check for hospital/clinic letterhead, proper medical symbols, and standard formatting.
                                               For invoices/receipts, verify company branding, standard layout, and expected design elements.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Design Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}

def Content_analysis_agent(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Individual_legitimacy_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""Analyze the provided document image based on the following indicator:
                            Content analysis: Assess the reasonableness and consistency of the actual content.
                                             For financial documents: Look for unusual transaction amounts (extremely small or round figures),
                                             suspicious account numbers, missing transaction fees or reference numbers.
                                             
                                             For medical documents: Check for inconsistent medical terminology, unrealistic
                                             treatment codes, missing provider information, or implausible diagnosis/treatment combinations.
                                             
                                             For invoices/receipts: Verify reasonable pricing, proper tax calculations,
                                             consistent product/service descriptions, and appropriate business details.
                                             
                                             For all documents: Look for suspicious or nonsensical text/descriptions,
                                             missing critical details that would normally be present in legitimate documents.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected Content Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning

    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""Analyze the provided document based on the following indicator:
                            Content analysis: Assess the reasonableness and consistency of the actual content.
                                             For financial documents: Look for unusual transaction amounts (extremely small or round figures),
                                             suspicious account numbers, missing transaction fees or reference numbers.
                                             
                                             For medical documents: Check for inconsistent medical terminology, unrealistic
                                             treatment codes, missing provider information, or implausible diagnosis/treatment combinations.
                                             
                                             For invoices/receipts: Verify reasonable pricing, proper tax calculations,
                                             consistent product/service descriptions, and appropriate business details.
                                             
                                             For all documents: Look for suspicious or nonsensical text/descriptions,
                                             missing critical details that would normally be present in legitimate documents.
                            
                            Determine with high accuracy if the document is 'fake' or 'legit'.
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain THIS SPECIFIC inconsistency type if detected: Content Issues, with specific examples and text cited from the document for THIS SPECIFIC issue.
                """}
        )

        response = model.with_structured_output(Individual_legitimacy_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your main task is to check if a document is fake based on the inconsistencies found. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
    
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING}


def Final_analysis(file, number_of_pages=1):
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")):
        image_path = str(file)
        image = Image.open(image_path)

        # Convert image to base64 for API transmission
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        response = model.with_structured_output(Final_analysis_Info).invoke(
            [
                SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your role is to do a final verification to check if a document is fake based on the inconsistencies found. Remember this task is crucial. The current date is {current_date}."),

                HumanMessage(content=[{"type": "text", "text": 
                            f"""The agent before you has tried some OCR and RAG to look for similar legitimate documents.
                            It has compared the input document to similar legitimate documents and found that the input document may be legit.
                            Your role is to do a final verification to determine if the document is fake or not. Remember this task is very crucial.

                            ## INSTRUCTIONS
                            Check the following points based on the document type:
                            
                            For financial documents:
                            - Date verification: Check if dates are logical and consistent
                            - Organization details: Verify bank/financial institution information
                            - Customer/account holder information
                            - Transaction details: Verify amounts, reference numbers, etc.
                            
                            For medical documents:
                            - Date verification: Check appointment/treatment dates
                            - Healthcare provider information
                            - Patient information
                            - Treatment/diagnosis details
                            
                            For invoices/receipts:
                            - Date verification
                            - Business/vendor information
                            - Customer information
                            - Product/service and pricing details
                            
                            For other documents:
                            - Date verification
                            - Issuing organization information
                            - Recipient information
                            - Content consistency and completeness

                            ## IMPORTANT NOTE
                            If you think that there are any inconsistencies in the information within the document, you should return that the document is fake.
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            Also return the forgery indicators (if you found any), else return None.

                            **BEFORE JUDGING, TRY TO UNDERSTAND THE WHOLE CONTEXT OF THE DOCUMENT.**
                            **DO NOT FOCUS ON MINOR ISSUES. IF THE OVERALL CONTEXT IS LEGITIMATE, THEN RETURN LEGIT.**
                            """
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    {"type": "text", "text": 
                        f"""
                        **Output Requirements:**
                        1.  Determine the overall `legitimacy` ('fake' or 'legit').
                        2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                        3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain the specific inconsistency types detected, with specific examples and text cited from the document for each issue.
                        4.  Also return the forgery indicators (if you found any), else return None.
                        """
                    }
                ]
                )
            ]
        )
        LEGITIMACY_REASONING = response.reasoning
        FORGERY_INDICATORS = response.forgery_indicators
    elif file.endswith(".pdf"):
        pdf_path = str(file)
        pdf_images = convert_from_path(pdf_path, dpi=300)

        # Limit the number of pages to process if needed
        pages_to_process = pdf_images[:number_of_pages]

        if not pages_to_process:
            print(f"  -> No pages extracted from PDF: {pdf_path}")
            # Handle error or return None/empty result
        message_content = [] # Start with an empty content list

        # Add the initial text prompt
        message_content.append(
            {"type": "text", "text":
                f"""The agent before you has tried some OCR and RAG to look for similar legitimate documents.
                            It has compared the input document to similar legitimate documents and found that the input document may be legit.
                            Your role is to do a final verification to determine if the document is fake or not. Remember this task is very crucial.

                            ## INSTRUCTIONS
                            Check the following points based on the document type:
                            
                            For financial documents:
                            - Date verification: Check if dates are logical and consistent
                            - Organization details: Verify bank/financial institution information
                            - Customer/account holder information
                            - Transaction details: Verify amounts, reference numbers, etc.
                            
                            For medical documents:
                            - Date verification: Check appointment/treatment dates
                            - Healthcare provider information
                            - Patient information
                            - Treatment/diagnosis details
                            
                            For invoices/receipts:
                            - Date verification
                            - Business/vendor information
                            - Customer information
                            - Product/service and pricing details
                            
                            For other documents:
                            - Date verification
                            - Issuing organization information
                            - Recipient information
                            - Content consistency and completeness

                            ## IMPORTANT NOTE
                            If you think that there are any inconsistencies in the information within the document, you should return that the document is fake.
                            Determine with high accuracy if the document is 'fake' or 'legit'.
                            Also return the forgery indicators (if you found any), else return None.

                            **BEFORE JUDGING, TRY TO UNDERSTAND THE WHOLE CONTEXT OF THE DOCUMENT.**
                            **DO NOT FOCUS ON MINOR ISSUES. IF THE OVERALL CONTEXT IS LEGITIMATE, THEN RETURN LEGIT.**
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

        message_content.append(
            {"type": "text", "text": 
                f"""
                Analyze the provided document pages based on the indicators listed previously.
                Determine if the document is 'fake' or 'legit'.
                
                **Output Requirements:**
                1.  Determine the overall `legitimacy` ('fake' or 'legit').
                2.  Provide a detailed `reasoning` string why the document is fake or legit. 
                3.  **IMPORTANT FOR REASONING:** If the document is fake, your reasoning **MUST** explain the specific inconsistency types detected, with specific examples and text cited from the document for each issue.
                4.  Also return the forgery indicators (if you found any), else return None.
                """}
        )

        response = model.with_structured_output(Final_analysis_Info).invoke([
            SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents. Your role is to do a final verification to check if a document is fake based on the inconsistencies found. Remember this task is crucial. The current date is {current_date}."),
            HumanMessage(content=message_content)
        ])
        LEGITIMACY_REASONING = response.reasoning
        FORGERY_INDICATORS = response.forgery_indicators
    if response.legitimacy == "fake":
        FAKE_DOC = True

    return {"FAKE_POT": FAKE_DOC, "LEGITIMACY_REASONING": LEGITIMACY_REASONING, "FAKE_INDICATOR": FORGERY_INDICATORS}
