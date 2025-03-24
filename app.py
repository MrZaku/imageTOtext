from flask import Flask, render_template, request
import cv2
import pytesseract
import os
from PIL import Image

app = Flask(__name__)

# Set Tesseract OCR Path (Windows users must specify the path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html', extracted_text="")

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)  # Save the uploaded image

    # Read the image using OpenCV
    image = cv2.imread(filepath)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to preprocess
    processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save preprocessed image (for debugging purposes)
    cv2.imwrite(filepath, processed_image)

    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string(Image.open(filepath))

    return render_template('index.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)
