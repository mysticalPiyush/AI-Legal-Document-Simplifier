# Install necessary libraries using pip
# - `transformers`: For pre-trained language models
# - `gradio`: To create a user-friendly interface
# - `PyPDF2`: To extract text from PDF files
!pip install transformers gradio PyPDF2

# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from PyPDF2 import PdfReader

# Specify the device for model execution (CPU is used here)
device = "cpu"

# Load a pre-trained tokenizer and language model (GPT-Neo 1.3B) from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)  # Move the model to CPU

def extract_text_from_pdf(file):
    """
    Reads and extracts text from a PDF file.
    - If the file contains no text or an error occurs, it handles the situation gracefully.
    """
    try:
        reader = PdfReader(file)  # Initialize a PDF reader
        text = ""
        # Loop through all pages and append their text content
        for page in reader.pages:
            text += page.extract_text()
        # Return None if no text is extracted
        if not text.strip():
            return None
        return text
    except Exception as e:
        # Return an error message if something goes wrong
        return f"Error extracting text: {e}"

def simplify_legal_document(file):
    """
    Simplifies a legal document uploaded as a PDF.
    - Converts complex legal language into plain English.
    - Highlights key obligations, penalties, and conditions.
    """
    try:
        # Step 1: Extract text from the PDF
        text = extract_text_from_pdf(file)
        if text is None:
            return "Could not extract text from the uploaded file. Please ensure it contains selectable text."
        if isinstance(text, str) and text.startswith("Error"):
            return text  # If there was an error during extraction, return it

        # Step 2: Prepare the prompt for the language model
        prompt = (
            "Simplify the following legal document into plain English and highlight key obligations, penalties, "
            "or conditions:\n\n" + text
        )

        # Tokenize the prompt and send it to the model for processing
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=200,  # Limit the output length
            num_return_sequences=1,  # Generate one response
            no_repeat_ngram_size=2  # Avoid repetitive phrases
        )

        # Decode the generated text into a human-readable format
        simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the simplified content by removing any unnecessary parts
        simplified_content = simplified_text.split("\n", 1)[-1]
        return simplified_content
    except Exception as e:
        # Handle and return any errors that occur during processing
        return f"Error during processing: {e}"

# Create a Gradio interface to interact with the tool
interface = gr.Interface(
    fn=simplify_legal_document,  # The function to simplify legal documents
    inputs=gr.File(label="Upload Legal Document (PDF)"),  # Accepts a file input (PDF)
    outputs=gr.Textbox(label="Simplified Document"),  # Outputs a simplified version of the text
    title="AI Legal Document Simplifier",  # Title for the web interface
    description=(
        "Upload a legal document (such as contracts, terms of service, or privacy policies) in PDF format, "
        "and this tool will simplify the language into plain English while highlighting key obligations, penalties, or conditions."
    ),
    examples=None,  # No example files provided for now
)

# Launch the Gradio app in a web browser
interface.launch()
