## 🤖 AI Financial Advisor

FinanceIQ now combines:
- **Data Analytics** (Pandas, Plotly)
- **Machine Learning** (Scikit-Learn Spending Prediction)
- **Hugging Face LLM** (Qwen2.5-3B-Instruct for natural language explanations)
- **Streamlit** (Interactive UI)

### Architecture
The LLM acts strictly as an **explanation and interaction layer**, not the numerical calculation engine:
1. Existing Python/ML code calculates verified financial metrics (income, expenses, predictions, health score).
2. These verified metrics are passed as a strict, sanitized context to the Hugging Face LLM.
3. The LLM generates natural language insights, action plans, and answers to user questions **without hallucinating numbers**.

### Local LLM Setup
The first time you run the app, it will download the `Qwen/Qwen2.5-3B-Instruct` model (approx. 6-8 GB). 
- **Minimum Requirement**: 8GB RAM (CPU execution will be slower).
- **Recommended**: A machine with a CUDA-compatible GPU for fast, responsive inference.
- **Privacy**: All processing happens locally. Your financial data is never sent to external cloud APIs.