import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_finance_llm():
    """Load the Qwen2.5-3B-Instruct model with resource caching."""
    try:
        model_id = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if device == "cpu":
            model = model.to(device)
            
        return tokenizer, model
    except torch.cuda.OutOfMemoryError:
        st.error("Unable to load FinanceIQ AI model: Insufficient RAM/VRAM. Please try a quantized model (e.g., AWQ) or a machine with more memory.")
        return None, None
    except Exception as e:
        st.error(f"Unable to load FinanceIQ AI model. Error: {str(e)}\nPlease check your Python environment and Hugging Face dependencies.")
        return None, None

def generate_financial_response(user_question: str, financial_context: str) -> str:
    """Generate a response using the LLM and verified financial context."""
    if not user_question or not user_question.strip():
        return "Please provide a valid question."
    
    tokenizer, model = load_finance_llm()
    if tokenizer is None or model is None:
        return "Unable to load FinanceIQ AI model. Please check your Python environment and Hugging Face dependencies."
    
    if not financial_context or not financial_context.strip():
        return "I don't have enough verified financial data to answer that accurately. Please provide or upload your income and expense information first."

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
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
    except torch.cuda.OutOfMemoryError:
        return "Error: Insufficient RAM/VRAM to run the model. Please try using a smaller or quantized model (e.g., Qwen2.5-3B-Instruct-AWQ), or run this on a machine with more memory."
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}."