# Bank Classification System

This project implements a robust **Bank Classification System** capable of identifying banks based on logos and text extracted from input documents (PDF or image files). It uses trained yolo model on custom dataset and OCR technologies for logo detection and text recognition, making it suitable for both scanned and digital documents.

## Features

- **PDF and Image Support**: Classifies banks from either PDF files or image files.
- **Scanned Document Detection**: Determines whether the input PDF is scanned or digital.
- **Logo Detection**: Uses a fine-tuned YOLO model for detecting bank logos.
- **Text Extraction**: Employs PaddleOCR for extracting text from detected logos.
- **Bank Name Classification**: Matches detected text with predefined bank names to classify the document.

## Technologies Used

- **YOLO**: For bank logo detection.
- **PaddleOCR**: For optical character recognition (OCR).
- **PyMuPDF (Fitz)**: To handle PDF documents.
- **pdf2image**: Converts PDF pages to images.
- **Streamlit**: Provides a user-friendly web interface.
- **Python Libraries**: `Pillow`, `numpy`, `torch`, `opencv-python`

## Setup Instructions

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or higher
- pip (Python package manager)
- GPU support (optional but recommended for faster YOLO model inference)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/bank-classification.git
   cd bank-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install additional tools for handling PDFs (if needed):
   ```bash
   sudo apt-get install poppler-utils
   ```

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Open the app in your browser (usually at `http://localhost:8501`).

3. Upload a PDF or image file, select the file type, and view the classification results.

## Code Overview

### `Banks_classification` Class

This class encapsulates the core functionality for bank classification:

- **Initialization**: Loads the model, input file (PDF or image), and OCR tool.
- **Scanned Document Detection**: Checks if a PDF contains selectable text.
- **Image Conversion**: Converts PDFs to images or preprocesses uploaded images.
- **Logo Detection**: Detects logos using YOLO.
- **Text Extraction**: Extracts text from detected logos using PaddleOCR.
- **Bank Classification**: Matches extracted text with predefined bank names.

### Streamlit Interface (`main.py`)

- Provides a simple user interface for uploading files and displaying results.
- Handles file uploads, model loading, and classification.

## Example Usage

1. Upload a PDF or image file.
2. The app detects logos and extracts text.
3. View the classified bank name and file type.

## Error Handling

The application includes robust error handling for scenarios such as:

- Invalid file uploads
- Missing model files
- Errors during PDF/image processing
- OCR extraction failures

## Future Enhancements

- Expand the predefined bank list for classification.
- Optimize YOLO and OCR models for better accuracy.
- Add support for multilingual text recognition.
- Enable batch processing of multiple files.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Contact**: For any questions or support, please reach out at `harshitasinghcal4839@gmail.com`.

