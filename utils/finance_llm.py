import streamlit as st
from huggingface_hub import InferenceClient

@st.cache_resource
def get_hf_client():
    """Initialize the Hugging Face Inference Client using Streamlit Secrets."""
    # Read the token from Streamlit's secure secrets manager
    token = st.secrets.get("HF_TOKEN", "")
    
    if not token or token == "your_huggingface_token_here":
        return None
        
    return InferenceClient(
        model="Qwen/Qwen2.5-1.5B-Instruct", # Small, fast, and highly capable
        token=token
    )

def generate_financial_response(user_question: str, financial_context: str) -> str:
    """Generate a response using the Hugging Face Free Inference API."""
    if not user_question or not user_question.strip():
        return "Please provide a valid question."
    
    client = get_hf_client()
    if client is None:
        return (
            "⚠️ **AI Model Not Configured**\n\n"
            "To enable the AI Advisor, please add your free Hugging Face Access Token "
            "to the Streamlit Secrets (Settings > Secrets). "
            "Get a free token at: https://huggingface.co/settings/tokens"
        )

    if not financial_context or not financial_context.strip():
        return "I don't have enough verified financial data to answer that accurately. Please upload your income and expense information first."

    system_prompt = """You are FinanceIQ AI, a personal finance education and explanation assistant.
Rules:
1. Never invent financial numbers. Use ONLY the verified financial context provided.
2. Do not modify calculated values. Treat Python/ML calculations as the absolute source of truth.
3. Do not pretend to be a licensed financial advisor.
4. Do not guarantee investment returns.
5. Clearly state when required data is missing.
6. Use simple, clear language and give practical, actionable recommendations.
7. Keep calculations outside the LLM.
8. If the user asks something unrelated to finance, politely steer them back to their financial data.
9. Output should be well-formatted, using bullet points where appropriate.
10. Use Indian Rupee (₹) formatting for any numbers mentioned."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Financial Context:\n{financial_context}\n\nUser Question: {user_question}"}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return (
            f"⚠️ **AI Service Temporarily Unavailable**\n\n"
            f"Error: {str(e)}\n\n"
            "This can happen if the free Hugging Face API is experiencing high traffic. "
            "Please try again in a moment, or use the '🧠 AI Budget Advisor' page for rule-based insights."
        )