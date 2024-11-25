# AI Legal Document Simplifier

## Overview
The **AI Legal Document Simplifier** is a tool designed to simplify legal documents, such as contracts, terms of service, or privacy policies. It leverages the power of GPT-Neo, a state-of-the-art language model, to convert complex legal jargon into plain English. Additionally, it highlights key obligations, penalties, or conditions to make legal content accessible to non-lawyers.

## Features
- **PDF Upload**: Users can upload a legal document in PDF format.
- **Text Simplification**: GPT-Neo processes the document to simplify the language.
- **Key Highlighting**: Highlights key obligations, penalties, or conditions.
- **User-Friendly**: Easy-to-use Gradio interface for interactive document simplification.

## Usefulness
This tool makes legal content more understandable for the general public, enabling better access to legal terms and conditions. It is especially helpful for non-lawyers who want to quickly comprehend complex legal documents.

## Technologies Used
- **GPT-Neo**: A powerful language model used to simplify and interpret legal texts.
- **Gradio**: A user-friendly interface for easy interaction with the model.
- **PyPDF2**: A Python library used to extract text from PDF files.
- **Torch**: A deep learning framework used to run the model.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10rbKTDHYpGftvSpjG4sG-kUzyqwWtIOD#scrollTo=aCqmOEnrL8sw)

## Setup and Installation

### Requirements:
- Python 3.7 or higher
- A Google Colab environment or local Python environment

### Install Dependencies:
To install the required dependencies, run the following command:

```bash
pip install transformers gradio PyPDF2
```

### Running the Project:

1. **Clone the repository** or open the code in Google Colab.
2. Install the required dependencies using the above command.
3. Upload a PDF file containing a legal document using the Gradio interface.
4. The tool will process the document and provide a simplified version in plain English with key points highlighted.

### Example Usage:

Once the app is running, you can interact with it as follows:

1. **Upload a legal document**: Click on the file input to upload your PDF.
2. **Simplify the document**: The model will automatically simplify the legal text.
3. **View the result**: The simplified document will be displayed in the output box.

### Sample Output:

- "This document is a contract between Company X and User Y. Key obligations include providing services by Company X and payments from User Y. Penalties for late payments include a fine of $100 per day."

## Running the Model on CPU

The model is designed to run on the CPU by default to avoid memory issues. However, if you have a GPU available, you can change the device setting in the code to use it for faster inference.

To run on the CPU, ensure that the `device` variable is set to `"cpu"`:

```python
device = "cpu"
```
