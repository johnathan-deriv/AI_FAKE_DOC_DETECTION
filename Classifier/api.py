import os
import tempfile
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from Document_classifier import Document_classifier # Import your main function

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Create this folder if it doesn't exist
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # 10 MB limit

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_base64(image_np):
    """Encodes an OpenCV image (NumPy array) to a Base64 string."""
    if image_np is None:
        return None
    try:
        # Encode image to JPEG format in memory
        is_success, buffer = cv2.imencode(".jpg", image_np)
        if not is_success:
            print("Error: Could not encode image to JPEG")
            return None
        # Encode the byte buffer to Base64 string
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

# --- NEW HELPER for JSON Serialization ---
def make_serializable(data):
    """Recursively converts NumPy types to standard Python types in a nested structure."""
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
        return int(data) # Convert numpy int types to python int
    elif isinstance(data, (np.float16, np.float32, np.float64)):
        return float(data) # Convert numpy float types to python float
    elif isinstance(data, (np.ndarray,)): # Handle numpy arrays (like points)
        return make_serializable(data.tolist()) # Convert array to list and recurse
    elif isinstance(data, (np.bool_)):
        return bool(data) # Convert numpy bool to python bool
    # Add other numpy types if needed (e.g., np.complex_)
    return data # Return unchanged if not a numpy type or nested structure

# --- API Route ---
@app.route('/classify', methods=['POST'])
def classify_document():
    """
    API endpoint to classify a document for forgery detection.
    Expects a POST request with a file ('file') and optional form data:
    - number_of_pages (int, default 1): For PDF analysis.
    - visualize (bool, default false): Whether to include annotated image(s).
    """
    # --- 1. Check for file part ---
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']

    # --- 2. Check for filename ---
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # --- 3. Validate file type and save securely ---
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_path = None
        try:
            # Use tempfile for secure temporary storage
            # Create a temporary file with the correct suffix
            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, dir=app.config['UPLOAD_FOLDER'], suffix=suffix) as temp_f:
                file.save(temp_f)
                temp_path = temp_f.name
            
            # --- 4. Get optional parameters ---
            try:
                num_pages = request.form.get('number_of_pages', default=1, type=int)
                if num_pages < 1:
                    num_pages = 1 # Ensure at least 1 page
            except ValueError:
                 return jsonify({'error': 'Invalid value for number_of_pages, must be an integer.'}), 400
            
            visualize = request.form.get('visualize', default='false').lower() == 'true'

            # --- 5. Call the Document Classifier ---
            print(f"Calling Document_classifier for {temp_path} (Pages: {num_pages}, Visualize: {visualize})")
            results = Document_classifier(temp_path, number_of_pages=num_pages, visualize_forgery=visualize)
            print("Document_classifier finished.")
            
            # --- 6. Process results for JSON response (handle images AND numpy types) ---
            # Encode annotated image(s) if they exist
            if visualize and results.get("FAKE_DOC"):
                if results.get("single_page") and "annotated_image" in results and results["annotated_image"] is not None:
                    print("Encoding single annotated image...")
                    results["annotated_image_base64"] = encode_image_base64(results["annotated_image"])
                    del results["annotated_image"] # Remove numpy array
                elif not results.get("single_page") and "annotated_images" in results and results["annotated_images"]:
                    print(f"Encoding {len(results['annotated_images'])} annotated images...")
                    results["annotated_images_base64"] = [encode_image_base64(img) for img in results["annotated_images"]]
                    del results["annotated_images"] # Remove list of numpy arrays
            
            # Remove potentially non-serializable image arrays if they weren't encoded
            if "annotated_image" in results and not isinstance(results["annotated_image"], (str, type(None))):
                 del results["annotated_image"]
            if "annotated_images" in results and not isinstance(results["annotated_images"], (list, type(None))):
                 del results["annotated_images"]

            # --- >>> NEW: Convert remaining results to be JSON serializable <<< ---
            serializable_results = make_serializable(results)
            # --- >>> END NEW <<< ---

            print("Returning JSON response.")
            return jsonify(serializable_results), 200 # Return the cleaned results

        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
        
        finally:
            # --- 7. Clean up temporary file ---
            if temp_path and os.path.exists(temp_path):
                print(f"Cleaning up temporary file: {temp_path}")
                os.unlink(temp_path)

    else:
        return jsonify({'error': 'File type not allowed'}), 400

# --- Main execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
