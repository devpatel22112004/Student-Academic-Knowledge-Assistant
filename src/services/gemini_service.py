import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_configured_api_key():
    """Read Gemini API key from .env file."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file. "
            "Please set your Gemini API key in .env file: GEMINI_API_KEY=your_key_here"
        )
    return api_key


def generate_with_flash(api_key, question, relevant_chunks):
    """Generate a grounded answer with a Gemini Flash model."""
    import google.generativeai as genai

    context = "\n\n".join(
        [f"Source: {chunk['source']}\nText: {chunk['text']}" for chunk in relevant_chunks]
    )

    prompt = f"""
You are an academic assistant. Answer only from the given context.

Rules:
1) If context has the answer, respond clearly in short bullet points.
2) If answer is missing, say: "I could not find this in your uploaded documents."
3) Do not add internet knowledge.

Question:
{question}

Context:
{context}
"""

    genai.configure(api_key=api_key)

    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash",
    ]

    available = []
    for model_info in genai.list_models():
        methods = getattr(model_info, "supported_generation_methods", [])
        if "generateContent" in methods:
            available.append(model_info.name)

    model_name = None
    for candidate in preferred:
        if candidate in available:
            model_name = candidate
            break

    if model_name is None:
        flash_models = [model for model in available if "flash" in model.lower()]
        if flash_models:
            model_name = flash_models[0]

    if model_name is None:
        raise RuntimeError("No supported Flash model is available for this API key.")

    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    if response and getattr(response, "text", ""):
        return response.text

    raise RuntimeError("Flash model returned an empty response.")