import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
from Document_classifier import Document_classifier, draw_boxes_on_image_cv2 # Import the main function and drawing utility
import io

# --- UI Configuration ---
st.set_page_config(layout="wide", page_title="Document Forgery Detector")
st.title("Document Forgery Detector")
st.write("Upload any document (Image or PDF) to check its legitimacy and visualize potential forgery indicators. Supports financial documents, medical bills, invoices, receipts, and more.")

# --- Helper Function for Legend ---
def display_legend(detected_indicator_types: set):
    """Displays a dynamic color legend for the detected bounding boxes."""
    if not detected_indicator_types:
        return # Don't show legend if nothing was detected/drawn

    st.subheader("Legend")

    # Use the keys consistent with analysis_result["bounding_boxes"][...]["indicator_type"]
    # Map keys to display names and colors
    color_map = {
        'customer_name': ("Customer Name", '#FF0000'), # Red
        'digital_artifacts': ("Digital Artifacts", '#00FF00'), # Lime Green
        'font_inconsistencies': ("Font Inconsistencies", '#0000FF'), # Blue
        'layout_issues': ("Layout Issues", '#FFFF00'), # Yellow
        'design_verification': ("Design Verification", '#FF00FF'), # Magenta
        'content_analysis': ("Content Analysis", '#FFA500'), # Orange
        'date_verification': ("Date Verification", '#800080'), # Purple
    }
    default_color = ("Unknown Indicator", '#808080') # Gray for fallback

    legend_html = "<div style='display: flex; flex-wrap: wrap;'>"
    for indicator_type in sorted(list(detected_indicator_types)):
        display_name, color_hex = color_map.get(indicator_type.lower(), default_color)

        legend_html += f"""
        <div style="margin-right: 20px; margin-bottom: 10px; display: flex; align-items: center;">
            <span style="background-color:{color_hex}; border-radius: 3px; width: 18px; height: 18px; display: inline-block; margin-right: 8px; border: 1px solid #ccc;"></span>
            <span>{display_name}</span>
        </div>
        """
    legend_html += "</div>"
    st.html(legend_html)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a file",
    type=["jpg", "jpeg", "png", "pdf"],
    help="Upload an image (JPG, PNG) or a PDF document."
)

# --- Main Logic ---
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    st.write("---")
    col1, col2 = st.columns([1, 1]) # Adjust column width ratio if needed

    # --- Column 1: Document Preview ---
    with col1:
        st.subheader("Uploaded Document Preview")
        try:
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption='Uploaded Image', use_container_width='auto')
            elif uploaded_file.type == "application/pdf":
                # Placeholder for PDF preview - could add pdf2image rendering if needed
                st.info(f"📄 PDF Uploaded: {uploaded_file.name}")
                # Optional: Display first page preview (requires pdf2image)
                # from pdf2image import convert_from_bytes
                # try:
                #     images = convert_from_bytes(uploaded_file.getvalue(), first_page=1, last_page=1, dpi=150)
                #     if images:
                #         st.image(images[0], caption='First Page Preview', use_column_width='auto')
                # except Exception as pdf_err:
                #     st.warning(f"Could not generate PDF preview: {pdf_err}")

            else:
                st.error("Unsupported file type for preview.")
        except Exception as preview_err:
            st.error(f"Error generating preview: {preview_err}")

    # --- Column 2: Analysis Control and Results ---
    with col2:
        st.subheader("Analysis")
        if st.button("Analyze Document", key="analyze_button", type="primary"):
            with st.spinner('🕵️ Analyzing document... This may take a moment.'):
                try:
                    # Call the classifier. Always request visualization.
                    # The Document_classifier function handles PDF page iteration internally if needed.
                    analysis_result = Document_classifier(temp_file_path, number_of_pages=1, visualize_forgery=True)

                    st.subheader("Analysis Results")

                    # Display Legitimacy Status
                    if analysis_result.get("FAKE_DOC"):
                        st.error("🚨 Document Classified as FAKE")
                    else:
                        st.success("✅ Document Classified as LEGITIMATE")

                    # Display Reasoning
                    reasoning = analysis_result.get('LEGITIMACY_REASONING', 'No reasoning provided.')
                    st.markdown(f"**Reasoning:** {reasoning}")

                    # Display Fake Indicators if present
                    indicators = analysis_result.get("FAKE_INDICATOR")
                    if indicators:
                         if isinstance(indicators, list):
                             st.markdown(f"**Indicators Found:** {', '.join(indicators)}")
                         elif isinstance(indicators, str):
                             st.markdown(f"**Indicator Found:** {indicators}")


                    # Display Similarity Info if present (from comparison branch)
                    if "SIMILAR_DOCUMENT_NAME" in analysis_result:
                        st.markdown("---")
                        st.markdown(f"**Compared Against:** `{analysis_result.get('SIMILAR_DOCUMENT_NAME', 'N/A')}`")
                        st.markdown(f"**Similarity Score:** {analysis_result.get('SIMILARITY_SCORE', 'N/A'):.2f}")
                        st.markdown(f"**Reference Legitimacy:** {'Legit' if analysis_result.get('SIMILAR_DOCUMENT_LEGITIMACY') else 'Fake'}")


                    # Display Annotated Image(s) if fake and available
                    if analysis_result.get("FAKE_DOC"):
                        annotated_image_found = False
                        if "annotated_image" in analysis_result and analysis_result["annotated_image"] is not None:
                             st.subheader("Forgery Visualization")
                             img_rgb = cv2.cvtColor(analysis_result["annotated_image"], cv2.COLOR_BGR2RGB)
                             st.image(img_rgb, caption='Annotated Forgery Indicators', use_container_width='auto')
                             annotated_image_found = True
                        elif "annotated_images" in analysis_result and analysis_result["annotated_images"]:
                             st.subheader("Forgery Visualization (PDF Pages)")
                             for i, img in enumerate(analysis_result["annotated_images"]):
                                 img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                 st.image(img_rgb, caption=f'Annotated Forgery Indicators - Page {i+1}', use_container_width='auto')
                             annotated_image_found = True

                        if annotated_image_found:
                            # Extract unique indicator types from the bounding boxes that were drawn
                            detected_types = set()
                            if "bounding_boxes" in analysis_result and analysis_result["bounding_boxes"]:
                                detected_types = set(box['indicator_type'] for box in analysis_result["bounding_boxes"])

                            display_legend(detected_types) # Pass detected types to the legend
                        else:
                            # Fallback visualization when no specific text matches are found
                            st.subheader("Forgery Visualization")
                            
                            # Create a simple visualization based on the forgery indicator
                            if uploaded_file.type.startswith('image'):
                                # For images, display with a colored border
                                img = Image.open(uploaded_file)
                                
                                # Create a colored border based on the forgery indicator
                                indicator = analysis_result.get("FAKE_INDICATOR", "")
                                border_color = "#FF0000"  # Default red
                                
                                if isinstance(indicator, str):
                                    if "digital_artifacts" in indicator.lower():
                                        border_color = "#00FF00"  # Green
                                    elif "font" in indicator.lower():
                                        border_color = "#0000FF"  # Blue
                                    elif "layout" in indicator.lower():
                                        border_color = "#FFFF00"  # Yellow
                                    elif "design" in indicator.lower():
                                        border_color = "#FF00FF"  # Magenta
                                    elif "content" in indicator.lower():
                                        border_color = "#FFA500"  # Orange
                                    elif "date" in indicator.lower():
                                        border_color = "#800080"  # Purple
                                
                                # Display the image with a colored border using HTML
                                st.markdown(
                                    f"""
                                    <div style="border:5px solid {border_color}; padding:10px; display: inline-block;">
                                        <p style="text-align:center; font-weight:bold; color:{border_color};">
                                            Suspicious Document - {indicator}
                                        </p>
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                                st.image(img, caption=f'Suspicious Document - {indicator}', use_container_width='auto')
                            
                            elif uploaded_file.type == "application/pdf":
                                # For PDFs, just display a warning with the indicator
                                indicator = analysis_result.get("FAKE_INDICATOR", "")
                                st.error(f"⚠️ This PDF contains suspicious elements: {indicator}")
                                st.info("Detailed visualization could not be generated. Please review the reasoning above for specific concerns.")
                            
                            # Create a simple legend based on the forgery indicator
                            indicator = analysis_result.get("FAKE_INDICATOR", "")
                            if indicator:
                                if isinstance(indicator, str):
                                    display_legend({indicator})
                                elif isinstance(indicator, list):
                                    display_legend(set(indicator))

                except ImportError as ie:
                     st.error(f"Import Error: {ie}. Make sure all required libraries (like 'pdf2image' for PDFs) are installed.")
                except FileNotFoundError as fnf:
                     st.error(f"File Not Found Error: {fnf}. This might indicate an issue with temporary file handling or external dependencies.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis:")
                    st.exception(e) # Shows the full traceback for debugging
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    st.success("Analysis complete.")
else:
    st.info("☝️ Upload a document using the browser above to begin analysis.")

st.write("---")
st.caption("Developed by AI Assistant | Powered by Streamlit")
