from PIL import Image, ImageFilter
from pdf2image import convert_from_path
import fitz  
import numpy as np
from ultralytics import YOLO
import cv2

class Banks_classification:
    def __init__(self, model, pdf=None, image=None, ocr=None):
        self.model = model
        self.pdf = pdf
        self.image = image
        self.ocr = ocr

    def is_scanned_pdf(self):
        if not self.pdf:
            return False
        
        doc = fitz.open(self.pdf)
        text = doc[0].get_text()
        doc.close()
        return not bool(text.strip())

    def convert_input_to_image(self):
        # If PDF is provided, convert first page to image
        if self.pdf:
            pages = convert_from_path(self.pdf, dpi=400, first_page=0, size=(640, 640))
            first_page = pages[0].convert('RGB')
            first_page = first_page.filter(ImageFilter.SHARPEN)
            return np.array(first_page)
        
        # If image is provided, load and resize it
        elif self.image:
            # Load image using PIL and convert to numpy array
            img = Image.open(self.image)
            img = img.resize((640, 640), Image.LANCZOS)
            img = img.convert('RGB')
            img = img.filter(ImageFilter.SHARPEN)
            return np.array(img)
        
        else:
            raise ValueError("No input file provided. Please supply either a PDF or an image.")
        
    def detect_logo(self, image):
        results = self.model.predict(source=image, imgsz=640, conf=0.4, iou=0.1)
        return results

    def extract_logo_bboxes(self, results):
        logo_bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls)]
                if label == "0":
                    logo_bboxes.append((x1-2, y1, x2+2, y2+2))
        return logo_bboxes

    def extract_logo_text(self, image, logo_bboxes):
        rx1, ry1, rx2, ry2 = logo_bboxes[0]
        if rx1 < rx2 and ry1 < ry2:
            cell_image = image[ry1:ry2, rx1:rx2]
            cell_image_pil = Image.fromarray(cell_image)
            text = ""
            try:
                result = self.ocr.ocr(np.array(cell_image_pil))
                if result:
                    for line in result[0]:
                        text += line[1][0]
                    text = text.replace(" ","").lower()
            except Exception as e:
                print(f"Error during OCR extraction: {e}")

            return text

    def classify_bank_ocr(self, bank_names):
        # Determine document type for PDF or Image
        if self.pdf:
            pdf_type = "Scanned" if self.is_scanned_pdf() else "Digital"
        elif self.image:
            # pdf_type = "Scanned" if self.is_scanned_image() else "Digital"
            pdf_type = "Scanned"
        else:
            return {"name": "Other", "type": "Unknown"}
        
        # Convert input to image
        try:
            image = self.convert_input_to_image()
        except ValueError as e:
            return {"name": "Other", "type": "Unknown"}

        # Detect logo
        result = self.detect_logo(image)
        logo_bbox = self.extract_logo_bboxes(result)
        print("Extracted bbox:", logo_bbox)

        # Process logo detection results
        if logo_bbox:
            extracted_text = self.extract_logo_text(image, logo_bbox)
            print(extracted_text)
            
            for bank_name in bank_names.keys():
                bank_words = bank_name.replace(" ","").lower()
                counter = 0
                if extracted_text == bank_words or bank_words in extracted_text or extracted_text in bank_words:
                    counter = counter + 1
                    if extracted_text == "natwest":
                        if logo_bbox[0][2] >= (640//2): #if the x2 value of logo bbox is greater than half of image width(half of 640)
                            return {"name": "Natwest", "type": pdf_type}
                        else:
                            return {"name": "NatwestOld", "type": pdf_type}
                        
                    elif extracted_text == "santanderonlinebanking":
                        return {"name": "Santander online banking", "type": pdf_type}
                    
                    return {"name": bank_names[bank_name], "type": pdf_type}
            
            if counter == 0:
                return {"name": "Other", "type": pdf_type}  
        else:
            return {"name": "Other", "type": pdf_type}