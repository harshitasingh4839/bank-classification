import streamlit as st
import numpy as np
from PIL import Image
import fitz
from paddleocr import PaddleOCR
import torch
import io
from ultralytics import YOLO
from bank_classification import Banks_classification

def load_models():
    # Load YOLO model for logo detection
    logo_model = YOLO('epoch88.pt')
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en', enable_mkdn = True)
    
    return logo_model, ocr

def main():
    st.title('Bank Classification')
    
    # Predefined bank names dictionary
    bank_names = {
        "Anna":"Anna",
        "Barclays":"Barclays", 
        "Capital on Tap":"Capital on Tap", 
        "CashPlus":"CashPlus", 
        "HSBC": "HSBC", 
        "Lloyds":"Lloyds", 
        "etro":"Metro",
        "Mettle":"Mettle",
        "Monzo":"Monzo", 
        "Nationwide Building Society":"Nationwide Building Society", 
        "Natwest":"Natwest", 
        "Revolut":"Revolut", 
        "Santander":"Santander",
        "Santander online Banking": "Santander online Banking",
        "starlingbank":"Starling Bank Account", 
        "Tide":"Tide", 
        "wise":"Transferwise"
        
    }

    # File type selection
    file_type = st.radio("Select file type:", ("PDF", "Image"))

    # File uploader based on selected type
    if file_type == "PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    else:
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:

        temp_file_path = "temp_upload." + uploaded_file.name.split('.')[-1]
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Load models
            logo_model, ocr = load_models()
            
            # Create Banks_classification instance
            bank_classifier = Banks_classification(
                model=logo_model, 
                # pdf="temp_upload.pdf", 
                pdf=temp_file_path if file_type == "PDF" else None,
                image=temp_file_path if file_type == "Image" else None,
                ocr=ocr
            )
            
            # Classify the bank
            result = bank_classifier.classify_bank_ocr(bank_names)
            
            # Display results
            st.subheader("Classification Results")
            st.write(f"Bank Name: {result['name']}")
            st.write(f"File Type: {result['type']}")
        
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")

if __name__ == "__main__":
    main()