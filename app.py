
import gradio as gr
import pytesseract
from PIL import Image

# Function to extract text using Tesseract OCR
def extract_text_from_image(image):
    # Use pytesseract to perform OCR on the image (assuming PIL image from Gradio)
    extracted_text = pytesseract.image_to_string(image, lang="eng+hin")  # 'eng+hin' for English and Hindi
    return extracted_text

# Function to search for keywords in the extracted text
def ocr_and_search(image, keyword):
    # Extract text from the uploaded image
    extracted_text = extract_text_from_image(image)
    
    # If a keyword is provided, highlight the keyword in the extracted text
    if keyword:
        highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
        return highlighted_text
    else:
        return extracted_text

# Gradio interface for uploading an image and searching for a keyword
interface = gr.Interface(
    fn=ocr_and_search,
    inputs=["image", "text"],
    outputs="text",
    title="OCR for Hindi and English Text",
    description="Upload an image with text in Hindi and English, and search for keywords."
)

# Launch the Gradio app
interface.launch()
