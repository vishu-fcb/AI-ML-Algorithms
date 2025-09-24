# 🧠 Smart Tech Assistant

A sleek and simple LLM-powered chatbot that lets you ask technical questions and get detailed markdown-formatted answers, powered by:

- ✅ OpenAI GPT-4o-mini (via API)
- ✅ Ollama LLaMA 3.2 (runs locally)
- 🎨 Fancy Gradio UI (with model selector and chat history)

---

## 🚀 Features

- Chat with OpenAI or a local Ollama model
- Markdown response formatting
- Custom system prompt (default: teaching assistant)
- Stylish UI with Gradio
- Works inside Jupyter Notebook (`inline=True`)

---

## 📦 Installation

1. **Clone the repo**:
   
   - git clone https://github.com/vishu-fcb/AI-ML-Algorithms.git
   - cd AI-ML-Algorithms\11_LLMs\AI_Technical_Assistant

2. **Create a virtual environment**:
    - python -m venv .venv
    - source .venv/bin/activate      # Linux/macOS
    - .venv\Scripts\activate         # Windows

3. **Install dependencies**:
    - pip install -r requirements.txt

4. **Set up .env file**:
    - Create an OPEN API KEY and save it in a .env file: https://platform.openai.com/docs/quickstart/create-and-export-an-api-key?api-mode=responses
    - Add your key to the .env file like this:
    - OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

5. **Running Ollama**:
    - https://ollama.com
    - Run in your terminal: ollama run llama3

6. **Run the code in Jupyter notebook**:
    - jupyter notebook



