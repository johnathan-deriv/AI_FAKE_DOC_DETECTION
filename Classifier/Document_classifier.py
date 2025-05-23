from agent_RAG import rag_agent
from agent_REAL_COMP import document_classifier_based_on_legit_docs
from agents_INDIVIDUAL_CHECK import Layout_issues_agent, Date_verification_agent, Design_verification_agent, Content_analysis_agent, Font_inconsistencies_agent, Digital_artifacts_agent, Final_analysis
from constants import model
from utils import forgery_text
from langchain.schema import HumanMessage, SystemMessage
from PIL import Image
import base64
import io
import cv2
from fuzzywuzzy import fuzz
import easyocr
import numpy as np


def get_text_and_boxes_ocr(image_path):
    """Uses EasyOCR to extract text and bounding boxes."""
    
    try:
        reader = easyocr.Reader(['en'], gpu=False) 
        
        # Read text from the image path
        ocr_data = reader.readtext(image_path, detail=1)
        
        results = []
        for (bbox, text, conf) in ocr_data:
            # bbox is a list of 4 points: [top_left, top_right, bottom_right, bottom_left]
            # Calculate center, width, height and angle from points
            
            # Convert polygon points to a rotated rectangle
            rect = cv2.minAreaRect(np.array(bbox, dtype=np.float32))
            # rect returns: ((center_x, center_y), (width, height), angle)
            (center_x, center_y), (width, height), angle = rect
            
            # Standard bbox coordinates for non-rotated rect (for compatibility)
            top_left = bbox[0]
            bottom_right = bbox[2]
            x = int(top_left[0])
            y = int(top_left[1])
            w = int(bottom_right[0] - top_left[0])
            h = int(bottom_right[1] - top_left[1])
            
            cleaned_text = text.strip()
            if cleaned_text: # Ignore empty strings
                results.append({
                    "text": cleaned_text,
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "rotated_box": {
                        "center": (int(center_x), int(center_y)),
                        "size": (int(width), int(height)),
                        "angle": angle,
                        "points": bbox  # Store the original 4 points
                    },
                    "confidence": conf
                })
                
        print(f"EasyOCR found {len(results)} text boxes.")
        return results
        
    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return []

def calculate_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def draw_boxes_on_image_cv2(image_path, list_of_boxes_to_draw, overlap_threshold=0.8):
    """Draws provided boxes onto the image using OpenCV, handling rotated boxes."""
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image for drawing: {image_path}")
        return None
        
    # Define colors with lowercase keys
    colors = {
        'customer_name': (0, 0, 255),         # Red
        'digital_artifacts': (0, 255, 0),     # Lime Green
        'font_inconsistencies': (255, 0, 0),  # Blue
        'layout_issues': (0, 255, 255),        # Yellow
        'design_verification': (255, 0, 255), # Magenta
        'content_analysis': (0, 165, 255),     # Orange
        'date_verification': (128, 0, 128),    # Purple 
    }
    
    drawn_boxes_coords = [] # Keep track of coordinates of boxes already drawn

    # Loop through the list of individual boxes to draw
    for box_info in list_of_boxes_to_draw:
        if box_info and 'box' in box_info and 'indicator_type' in box_info:
            indicator_type = box_info['indicator_type']
            lookup_key = indicator_type.lower()
            color = colors.get(lookup_key, (128, 128, 128)) # Default gray
            
            # Check if we have rotated box information inside the 'box' object
            if 'box' in box_info and 'rotated_box' in box_info['box']:
                print(f"Drawing rotated box for: {box_info.get('text', '')}")
                rotated_box = box_info['box']['rotated_box']
                
                # Get the points from the rotated_box
                if 'points' in rotated_box:
                    # Convert points to proper array format for contours
                    points = rotated_box['points']
                    # Convert numpy float64 values to integers if needed
                    box_points = []
                    for point in points:
                        # Handle both list and numpy array formats
                        if hasattr(point, '__iter__'):  # Check if it's iterable
                            box_points.append([int(float(point[0])), int(float(point[1]))])
                        else:
                            # Fallback in case point is not iterable
                            print(f"Warning: Unexpected point format: {point}, type: {type(point)}")
                            continue
                    
                    # If we have valid points, draw the contour
                    if box_points and len(box_points) >= 3:  # Need at least 3 points for a polygon
                        box_points_array = np.array(box_points, dtype=np.int32)
                        cv2.drawContours(image, [box_points_array], 0, color, 3)
                        print(f"Drew rotated box with points: {box_points}")
                        drawn_boxes_coords.append(True)  # Mark as drawn
                    else:
                        print(f"Not enough valid points for rotated box: {box_points}")
                else:
                    print(f"No 'points' in rotated_box: {rotated_box}")
                    
            # If no valid rotated box, use standard rectangle instead
            else:
                box_data = box_info['box']
                x, y, w, h = box_data['x'], box_data['y'], box_data['width'], box_data['height']
                current_box_coords = (x, y, x + w, y + h)
                 
                # Overlap Check
                is_overlapping = False
                for drawn_coords in drawn_boxes_coords:
                    if isinstance(drawn_coords, tuple):  # Skip if it's not a rectangle coordinate
                        iou = calculate_iou(current_box_coords, drawn_coords)
                        if iou > overlap_threshold:
                            print(f"Skipping box for '{box_info.get('text', indicator_type)}' due to overlap")
                            is_overlapping = True
                            break
                 
                if is_overlapping:
                    continue
                
                # Draw standard rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
                drawn_boxes_coords.append(current_box_coords)

        else:
            print(f"Skipping invalid box info: {box_info}")

    return image

def find_box_for_text(ocr_results, target_text, threshold=85):
    """
    Finds the bounding box from OCR results that best matches the target text
    using fuzzy matching.
    """
    if not target_text:
        return None

    best_match_box = None
    best_match_item = None
    highest_similarity_score = 0

    # Clean the target text slightly (remove common punctuation that might confuse)
    cleaned_target = target_text.strip().replace(':', '').replace(',', '')

    for item in ocr_results:
        ocr_text = item['text']
        cleaned_ocr = ocr_text.strip().replace(':', '').replace(',', '')
        
        # Calculate similarity score (adjust ratio type if needed, WRatio is often good)
        # Consider using fuzz.token_set_ratio for multi-word robustness
        similarity_score = fuzz.WRatio(cleaned_target.lower(), cleaned_ocr.lower()) # Case-insensitive match

        # Track the best match above the threshold
        if similarity_score > highest_similarity_score and similarity_score >= threshold:
            highest_similarity_score = similarity_score
            best_match_box = item['box']
            best_match_item = item
            
            # Optional: If we get a perfect score, maybe return immediately
            if similarity_score == 100:
                 print(f"  -> Found perfect fuzzy match for '{target_text}' ({ocr_text}) score={similarity_score}")
                 # Create a fresh copy of the box
                 result_box = best_match_box.copy()
                 # Add rotated box information if available
                 if 'rotated_box' in best_match_item:
                     result_box['rotated_box'] = best_match_item['rotated_box']
                 return result_box

    if best_match_box:
        print(f"  -> Found best fuzzy match for '{target_text}' score={highest_similarity_score}")
        # Create a fresh copy of the box
        result_box = best_match_box.copy()
        # Add rotated box information if available
        if best_match_item and 'rotated_box' in best_match_item:
            result_box['rotated_box'] = best_match_item['rotated_box']
        return result_box
    else:
        print(f"  -> No fuzzy match found above threshold {threshold} for '{target_text}'.")
        return None # No suitable box found

def get_fake_text_from_reasoning(image_path, reasoning, indicators):
    #load the image
    image = Image.open(image_path)
    #draw the bounding boxes
    buffer = io.BytesIO()
    image.save(buffer, format=image.format or "JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = model.with_structured_output(forgery_text).invoke(
                [
                    SystemMessage(content=f"You are analyzing various types of documents to detect potential forgeries. These could include financial documents, medical bills, invoices, receipts, or any other official documents."),

                    HumanMessage(content=[{"type": "text", "text": 
                                f"""Here is the forgery set of indicators found in the document: 
                                {indicators}
                                Here is the reasoning for why the document is fake: 
                                {reasoning}

                                Please extract the specific text from the image that corresponds EXACTLY to the forgery signs pointed out in the reasoning.
                                
                                **SPECIAL INSTRUCTIONS:**
                                *   For indicators like 'digital_artifacts' or 'font_inconsistencies', extract the exact number, word, or phrase mentioned in the reasoning as showing the issue.
                                *   For indicators like 'design_verification' or 'layout_issues' that might relate to visual elements or logos:
                                    *   If the reasoning specifically mentions text (e.g., "the logo text is wrong"), extract that exact text (e.g., "Logo Text").
                                    *   If the reasoning refers to a visual element without specific text (e.g., "logo is misplaced", "layout is inconsistent"), extract the nearest descriptive text label OR the section title (e.g., if layout issue is in 'Details', extract 'Details'). If no relevant text is nearby, return null or an empty string for that indicator.
                                *   If the reasoning mentions multiple text elements for one indicator, list them separated by a newline character ('\\n').

                                return the result as a json object matching the 'forgery_text' schema.

                                Example 1 Output Format if only one indicator is found:
                                {{
                                    "Digital_artifacts": "0777118293"
                                }}

                                Example 2 Output Format if only one indicator is found:
                                {{
                                    "Font_inconsistencies": "Transaction ID\\n0776993754"
                                }}

                                Example 3 Output Format if multiple indicators are found:
                                {{
                                    "Digital_artifacts": "0777118293",
                                    "Font_inconsistencies": "Transaction ID\\n0776993754",
                                    "Design_verification": "Logo",
                                    "Layout_issues": "Details"
                                }}
                                ...
                                """

                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{base64_image}"}},
                    ]
                    )
                ]
            )
    return response


def Individual_analysis(file, number_of_pages=1):
    # Initialize these variables at the beginning to avoid UnboundLocalError
    FAKE_DOC = False
    LEGITIMACY_REASONING = None
    FAKE_INDICATOR = None
    
    print("Entering Digital Artifacts agent")
    digital_artifacts_result = Digital_artifacts_agent(file, number_of_pages=number_of_pages)
    if digital_artifacts_result["FAKE_POT"]:
        FAKE_DOC = True
        LEGITIMACY_REASONING = digital_artifacts_result["LEGITIMACY_REASONING"]
        FAKE_INDICATOR = 'Digital_artifacts'
    else:
        print("Digital Artifacts agent found no fake document")
        print("Entering Font Inconsistencies agent")
        font_inconsistencies_result = Font_inconsistencies_agent(file, number_of_pages=number_of_pages)
        if font_inconsistencies_result["FAKE_POT"]:
            FAKE_DOC = True
            LEGITIMACY_REASONING = font_inconsistencies_result["LEGITIMACY_REASONING"]
            FAKE_INDICATOR = 'Font_inconsistencies'
        else:
            print("Font Inconsistencies agent found no fake document")
            print("Entering Layout Issues agent")
            layout_issues_result = Layout_issues_agent(file, number_of_pages=number_of_pages)
            if layout_issues_result["FAKE_POT"]:
                FAKE_DOC = True
                LEGITIMACY_REASONING = layout_issues_result["LEGITIMACY_REASONING"]
                FAKE_INDICATOR = 'Layout_issues'
            else:
                print("Layout Issues agent found no fake document")
                print("Entering Design Verification agent")
                design_verification_result = Design_verification_agent(file, number_of_pages=number_of_pages)
                if design_verification_result["FAKE_POT"]:
                    FAKE_DOC = True
                    LEGITIMACY_REASONING = design_verification_result["LEGITIMACY_REASONING"]
                    FAKE_INDICATOR = 'Design_verification'
                else:
                    print("Design Verification agent found no fake document")
                    print("Entering Content Analysis agent")
                    content_analysis_result = Content_analysis_agent(file, number_of_pages=number_of_pages)
                    if content_analysis_result["FAKE_POT"]:
                        FAKE_DOC = True
                        LEGITIMACY_REASONING = content_analysis_result["LEGITIMACY_REASONING"]
                        FAKE_INDICATOR = 'Content_analysis'
                    else:
                        print("Content Analysis agent found no fake document")
                        print("Entering Date Verification agent")
                        date_verification_result = Date_verification_agent(file, number_of_pages=number_of_pages)
                        if date_verification_result["FAKE_POT"]:
                            FAKE_DOC = True
                            LEGITIMACY_REASONING = date_verification_result["LEGITIMACY_REASONING"]
                            FAKE_INDICATOR = 'Date_verification'
    #if no fake document found, then it is legit
    if not FAKE_DOC:
            LEGITIMACY_REASONING = "Document passed all individual checks."
            FAKE_INDICATOR = None # Or an empty string/list
    result = {"FAKE_DOC": FAKE_DOC,
                "LEGITIMACY_REASONING": LEGITIMACY_REASONING,
                "FAKE_INDICATOR": FAKE_INDICATOR}
    return result


def Document_classifier(file, number_of_pages=1, visualize_forgery=False):
    FAKE_DOC = False

    similar_legit_docs = rag_agent(file, number_of_pages)
    for doc in similar_legit_docs:
        print(doc[1])
    # filter down docs to score above 0.64
    similar_legit_docs = [doc for doc in similar_legit_docs if doc[1] > 0.64]
    # Check if the similar_docs is None or empty
    if similar_legit_docs is None or len(similar_legit_docs) == 0:
        print("No similar documents found, running individual analysis")
        # Running individual analysis agents with order of priority (quit if anyone is fake)
        # 1. Digital Artifacts
        # 2. Font Inconsistencies
        # 3. Layout Issues
        # 4. Design Verification
        # 5. Content Analysis
        # 6. Date Verification
        result = Individual_analysis(file, number_of_pages=number_of_pages)
        
    
    else:
        print(f"Found {len(similar_legit_docs)} similar documents, running similarity analysis")
        for doc in similar_legit_docs:
            #print similarity score
            print(f"Similarity score: {doc[1]}")
        Legit_doc_comparison = document_classifier_based_on_legit_docs(file, similar_legit_docs, number_of_pages=number_of_pages)
        if Legit_doc_comparison["REAL_DOC"]:
            # run individual analysis on the similar document
            result = Final_analysis(file, number_of_pages=number_of_pages)
        else:
            FAKE_DOC = True
            LEGITIMACY_REASONING = Legit_doc_comparison["LEGITIMACY_REASONING"]
            SIMILAR_DOCUMENT_NAME = Legit_doc_comparison["SIMILAR_DOCUMENT_NAME"]
            SIMILAR_DOCUMENT_LEGITIMACY = Legit_doc_comparison["SIMILAR_DOCUMENT_LEGITIMACY"]
            SIMILARITY_SCORE = Legit_doc_comparison["SIMILARITY_SCORE"]
            FAKE_INDICATOR = Legit_doc_comparison["FAKE_INDICATORS"]
            result = {"FAKE_DOC": FAKE_DOC,
                    "LEGITIMACY_REASONING": LEGITIMACY_REASONING,
                    "SIMILAR_DOCUMENT_NAME": SIMILAR_DOCUMENT_NAME,
                    "SIMILAR_DOCUMENT_LEGITIMACY": SIMILAR_DOCUMENT_LEGITIMACY,
                    "SIMILARITY_SCORE": SIMILARITY_SCORE,
                    "FAKE_INDICATOR": FAKE_INDICATOR} # Ensure correct key is used
              
    # ==== VISUALIZATION CODE FOR PDF AND IMAGE SUPPORT ====
    if visualize_forgery and result["FAKE_DOC"]:
        print("Attempting to visualize forgery...")
        
        # Initialize lists to store results for multiple pages
        all_annotated_images = []  # Will store all annotated page images
        all_boxes_found = []  # Will store bounding boxes for all pages
        
        # Check if the file is a PDF or an image
        if file.lower().endswith('.pdf'):
            print(f"Processing PDF file with {number_of_pages} pages for visualization")
            import tempfile
            from pdf2image import convert_from_path
            
            try:
                # Convert PDF pages to images (up to the specified number_of_pages)
                pdf_images = convert_from_path(file, dpi=300, first_page=1, last_page=number_of_pages)
                
                # Create a temporary directory to store individual page images
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_page_files = []
                    
                    # Save each PDF page as an image file
                    for i, page_image in enumerate(pdf_images):
                        page_file_path = f"{temp_dir}/page_{i+1}.jpg"
                        page_image.save(page_file_path, "JPEG")
                        temp_page_files.append(page_file_path)
                    
                    # Process each page image
                    for page_idx, page_file_path in enumerate(temp_page_files):
                        print(f"Processing PDF page {page_idx+1}/{len(temp_page_files)}")
                        
                        # Get forgery text for this page
                        try:
                            # Get forged text from reasoning - we're using the same reasoning for all pages
                            # We might want to make this page-specific in a more advanced version
                            forged_text_map = get_fake_text_from_reasoning(page_file_path, result["LEGITIMACY_REASONING"], result["FAKE_INDICATOR"])
                            
                            # Initialize for this page
                            page_boxes_found = []
                            page_annotated_image = None
                            
                            if forged_text_map:
                                print(f"Page {page_idx+1}: Identified forged text map: {forged_text_map}")
                                ocr_results = get_text_and_boxes_ocr(page_file_path)
                                
                                if ocr_results:
                                    print(f"Page {page_idx+1}: Matching forged text to OCR boxes...")
                                    matched_coords_set = set()
                                    
                                    # Same box-finding logic as before, but for this page
                                    for indicator_type, combined_target_text in forged_text_map.__dict__.items():
                                        if combined_target_text:
                                            individual_target_texts = combined_target_text.split('\n')
                                            for target_text in individual_target_texts:
                                                target_text = target_text.strip()
                                                if not target_text: continue
                                                
                                                found_box = find_box_for_text(ocr_results, target_text)
                                                
                                                if found_box:
                                                    # Store the coordinates for overlap detection
                                                    box_tuple = (found_box['x'], found_box['y'], found_box['width'], found_box['height'])
                                                    
                                                    if box_tuple not in matched_coords_set:
                                                        print(f"Page {page_idx+1}: Found unique box for '{target_text}': {found_box}")
                                                        # Create a new box_info that includes all the data
                                                        box_info = {
                                                            "indicator_type": indicator_type,
                                                            "text": target_text,
                                                            "box": found_box,  # This now includes rotated_box if available
                                                            "page": page_idx+1  # Add page number to the box info
                                                        }
                                                        page_boxes_found.append(box_info)
                                                        matched_coords_set.add(box_tuple)
                                    
                                    # Draw boxes for this page
                                    if page_boxes_found:
                                        print(f"Page {page_idx+1}: Drawing {len(page_boxes_found)} non-overlapping boxes...")
                                        page_annotated_image = draw_boxes_on_image_cv2(page_file_path, page_boxes_found)
                            
                            # Store results for this page
                            if page_annotated_image is not None:
                                all_annotated_images.append(page_annotated_image)
                            if page_boxes_found:
                                all_boxes_found.extend(page_boxes_found)
                                
                        except Exception as e:
                            print(f"Error processing PDF page {page_idx+1}: {e}")
                
                # Update result with PDF-specific visualization data
                if all_annotated_images:
                    result["annotated_images"] = all_annotated_images  # Multiple page images
                    result["single_page"] = False  # Flag indicating multiple pages
                if all_boxes_found:
                    result["bounding_boxes"] = all_boxes_found  # All bounding boxes with page info
                
                # If no visualization was created, create a fallback visualization for PDF
                if not all_annotated_images:
                    try:
                        # Create a simple colored border around each page
                        fallback_images = []
                        
                        for i, page_image in enumerate(pdf_images):
                            # Convert PIL Image to OpenCV format
                            page_file_path = f"{temp_dir}/page_{i+1}.jpg"
                            image = cv2.imread(page_file_path)
                            
                            if image is not None:
                                # Determine border color based on indicator type
                                indicator = result.get("FAKE_INDICATOR", "").lower()
                                border_color = (0, 0, 255)  # Default red
                                
                                if "digital_artifacts" in indicator:
                                    border_color = (0, 255, 0)  # Green
                                elif "font_inconsistencies" in indicator:
                                    border_color = (255, 0, 0)  # Blue
                                elif "layout_issues" in indicator:
                                    border_color = (0, 255, 255)  # Yellow
                                elif "design_verification" in indicator:
                                    border_color = (255, 0, 255)  # Magenta
                                elif "content_analysis" in indicator:
                                    border_color = (0, 165, 255)  # Orange
                                elif "date_verification" in indicator:
                                    border_color = (128, 0, 128)  # Purple
                                
                                # Add a colored border
                                border_size = 20
                                bordered_image = cv2.copyMakeBorder(
                                    image, 
                                    border_size, border_size, border_size, border_size, 
                                    cv2.BORDER_CONSTANT, 
                                    value=border_color
                                )
                                
                                # Add text indicating the forgery type
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                text = f"Suspicious: {result.get('FAKE_INDICATOR', '')} (Page {i+1})"
                                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                                text_x = (bordered_image.shape[1] - text_size[0]) // 2
                                cv2.putText(
                                    bordered_image, 
                                    text, 
                                    (text_x, 30), 
                                    font, 1, 
                                    border_color, 
                                    2
                                )
                                
                                fallback_images.append(bordered_image)
                        
                        if fallback_images:
                            # Store the fallback visualizations
                            result["annotated_images"] = fallback_images
                            result["single_page"] = False
                            
                            # Create dummy bounding boxes for the legend
                            dummy_boxes = []
                            for i in range(len(fallback_images)):
                                dummy_box = {
                                    "indicator_type": result.get("FAKE_INDICATOR", ""),
                                    "text": "Suspicious Document",
                                    "box": {"x": 0, "y": 0, "width": 100, "height": 100},
                                    "page": i+1
                                }
                                dummy_boxes.append(dummy_box)
                            result["bounding_boxes"] = dummy_boxes
                    except Exception as e:
                        print(f"Failed to create fallback visualization for PDF: {e}")
                
            except Exception as e:
                print(f"Error during PDF visualization: {e}")
        
        else:
            # Original single-image handling code
            print("Processing single image file for visualization")
            annotated_opencv_image = None
            boxes_found_for_drawing = []
            
            try:
                forged_text_map = get_fake_text_from_reasoning(file, result["LEGITIMACY_REASONING"], result["FAKE_INDICATOR"])
                if not forged_text_map:
                    print("Could not extract specific forged text from reasoning.")
                else:
                    print(f"Identified forged text map: {forged_text_map}")
                    ocr_results = get_text_and_boxes_ocr(file)
                    if not ocr_results:
                        print("OCR failed or found no text.")
                    else:
                        print("Matching forged text to OCR boxes...")
                        matched_coords_set = set()
                        
                        for indicator_type, combined_target_text in forged_text_map.__dict__.items():
                            if combined_target_text:
                                individual_target_texts = combined_target_text.split('\n')
                                for target_text in individual_target_texts:
                                    target_text = target_text.strip()
                                    if not target_text: continue
                                    
                                    print(f"Searching for box matching '{target_text}' for indicator '{indicator_type}'")
                                    found_box = find_box_for_text(ocr_results, target_text)
                                    
                                    if found_box:
                                        # Store the coordinates for overlap detection
                                        box_tuple = (found_box['x'], found_box['y'], found_box['width'], found_box['height'])
                                        
                                        if box_tuple not in matched_coords_set:
                                            print(f"Found unique box for '{target_text}': {found_box}")
                                            # Create a new box_info that includes all the data
                                            box_info = {
                                                "indicator_type": indicator_type,
                                                "text": target_text,
                                                "box": found_box,  # This now includes rotated_box if available
                                                "page": 1  # Single image is treated as page 1
                                            }
                                            boxes_found_for_drawing.append(box_info)
                                            matched_coords_set.add(box_tuple)
                                        else:
                                            print(f"Skipping box for '{target_text}' as coordinates {box_tuple} already used by another indicator.")
                                    else:
                                        print(f"Could not find matching OCR box for: '{target_text}'")
                            else:
                                print(f"No target text for indicator: {indicator_type}")

                        if boxes_found_for_drawing:
                            print(f"Drawing {len(boxes_found_for_drawing)} non-overlapping boxes...")
                            annotated_opencv_image = draw_boxes_on_image_cv2(file, boxes_found_for_drawing)
                            if annotated_opencv_image is None:
                                print("Drawing failed, could not create annotated image.")
                            else:
                                print("Annotated image created successfully (in memory).")
                        else:
                            print("No unique boxes found to draw.")

            except Exception as e:
                print(f"Error during visualization steps: {e}")

            # Update result with single-image visualization data
            if annotated_opencv_image is not None:
                result["annotated_image"] = annotated_opencv_image  # Single image
                result["single_page"] = True  # Flag indicating single page
            if boxes_found_for_drawing:
                result["bounding_boxes"] = boxes_found_for_drawing  # Bounding boxes for single image
            
            # If no visualization was created, create a fallback visualization
            if annotated_opencv_image is None:
                try:
                    # Create a simple colored border around the image
                    image = cv2.imread(file)
                    if image is not None:
                        # Determine border color based on indicator type
                        indicator = result.get("FAKE_INDICATOR", "").lower()
                        border_color = (0, 0, 255)  # Default red
                        
                        if "digital_artifacts" in indicator:
                            border_color = (0, 255, 0)  # Green
                        elif "font_inconsistencies" in indicator:
                            border_color = (255, 0, 0)  # Blue
                        elif "layout_issues" in indicator:
                            border_color = (0, 255, 255)  # Yellow
                        elif "design_verification" in indicator:
                            border_color = (255, 0, 255)  # Magenta
                        elif "content_analysis" in indicator:
                            border_color = (0, 165, 255)  # Orange
                        elif "date_verification" in indicator:
                            border_color = (128, 0, 128)  # Purple
                        
                        # Add a colored border
                        border_size = 20
                        bordered_image = cv2.copyMakeBorder(
                            image, 
                            border_size, border_size, border_size, border_size, 
                            cv2.BORDER_CONSTANT, 
                            value=border_color
                        )
                        
                        # Add text indicating the forgery type
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = f"Suspicious: {result.get('FAKE_INDICATOR', '')}"
                        text_size = cv2.getTextSize(text, font, 1, 2)[0]
                        text_x = (bordered_image.shape[1] - text_size[0]) // 2
                        cv2.putText(
                            bordered_image, 
                            text, 
                            (text_x, 30), 
                            font, 1, 
                            border_color, 
                            2
                        )
                        
                        # Store the fallback visualization
                        result["annotated_image"] = bordered_image
                        result["single_page"] = True
                        
                        # Create a dummy bounding box for the legend
                        dummy_box = {
                            "indicator_type": result.get("FAKE_INDICATOR", ""),
                            "text": "Suspicious Document",
                            "box": {"x": 0, "y": 0, "width": 100, "height": 100},
                            "page": 1
                        }
                        result["bounding_boxes"] = [dummy_box]
                except Exception as e:
                    print(f"Failed to create fallback visualization: {e}")

    # Convert FAKE_POT to FAKE_DOC in the result if it exists
    if "FAKE_POT" in result:
        result["FAKE_DOC"] = result.pop("FAKE_POT")

    return result
