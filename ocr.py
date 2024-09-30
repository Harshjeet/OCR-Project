# # ocr_model.py

# from transformers import pipeline
# token = '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'

# # Initialize the model for OCR
# def load_ocr_model():
#     ocr_model = pipeline("automatic-speech-recognition", model="ColPali/Byaldi-Qwen2-VL" , use_outh_token = token)
#     return ocr_model

# def extract_text_from_image(image_path):
#     ocr_model = load_ocr_model()
#     # Pass the image directly to the model
#     with open(image_path, 'rb') as image:
#         result = ocr_model(image)
#     return result['text']

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the processor and model from Huggingface
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def extract_text_from_image(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Preprocess the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate the predicted text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text
