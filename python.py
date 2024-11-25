# Install necessary libraries
!pip install transformers gradio PyPDF2

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from PyPDF2 import PdfReader

# Use CPU for model execution (set device to 'cpu')
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)  # Move model to CPU

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file.
    """
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            return None  # Handle cases where no text is extracted
        return text
    except Exception as e:
        return f"Error extracting text: {e}"

def simplify_legal_document(file):
    """
    Simplifies the uploaded legal document and highlights key obligations, penalties, or conditions.
    """
    try:
        # Extract text from the uploaded PDF file
        text = extract_text_from_pdf(file)
        if text is None:
            return "Could not extract text from the uploaded file. Please ensure it contains selectable text."
        if isinstance(text, str) and text.startswith("Error"):
            return text  # Return extraction error message

        # Simplify the legal document
        prompt = (
            "Simplify the following legal document into plain English and highlight key obligations, penalties, "
            "or conditions:\n\n" + text
        )

        # Tokenize and move inputs to the appropriate device (CPU)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=200,  # Number of tokens to generate
            num_return_sequences=1, 
            no_repeat_ngram_size=2
        )

        # Decode the output text
        simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract simplified content and remove prompt
        simplified_content = simplified_text.split("\n", 1)[-1]
        return simplified_content
    except Exception as e:
        return f"Error during processing: {e}"

# Gradio Interface
interface = gr.Interface(
    fn=simplify_legal_document,
    inputs=gr.File(label="Upload Legal Document (PDF)"),
    outputs=gr.Textbox(label="Simplified Document"),
    title="AI Legal Document Simplifier",
    description=(
        "Upload a legal document (such as contracts, terms of service, or privacy policies) in PDF format, "
        "and this tool will simplify the language into plain English while highlighting key obligations, penalties, or conditions."
    ),
    examples=None,
)

# Launch the Gradio App
interface.launch()
