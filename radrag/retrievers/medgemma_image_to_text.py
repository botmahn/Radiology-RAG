from langsmith import Client
from langsmith.run_helpers import traceable

from typing import Dict, Any
import ollama
import os

os.environ["HF_HOME"] = "/Users/namanmishra/Documents/Code/iiith_courses/lma/major_project/Radiology-RAG/pretrained_models"

@traceable(name="generate_initial_diagnosis")
def generate_initial_diagnosis(
    image_base64: str,
    patient_details: str,
    model: str = "amsaravi/medgemma-4b-it:q8"
) -> str:

    prompt = f"""You are an expert radiologist assistant. Analyze the given chest X-ray image and the following patient details and the indications to generate a short diagnosis.

Patient Details and Indications:
{patient_details}

Provide only 3-4 lines of a concise initial assessment focusing on the most significant findings. 
If the provided image is not an X-ray image or if the patient details do not pertain to a medical use-case, return 'invalid'."""

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }]
        )
        
        isd = response['message']['content'].strip()
        return isd
    
    except Exception as e:
        print(f"Error generating initial diagnosis: {e}")
        return "Unable to generate initial diagnosis. Please check model availability."