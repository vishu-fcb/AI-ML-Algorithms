# AI Technical Assistant

> A dual-model technical Q&A chatbot — switch between OpenAI GPT-4o-mini (cloud) and locally-running Ollama LLaMA 3, all from a polished Gradio interface.

---

## Overview

This project demonstrates how to build a **model-agnostic LLM chat interface** that works with both cloud and local models. It is packaged as a Jupyter Notebook with an inline Gradio UI.

Use it as a starting point for any Q&A assistant — swap the system prompt to change the assistant's domain expertise.

---

## Features

- Chat with **OpenAI GPT-4o-mini** (fast, cloud-based) or **Ollama LLaMA 3** (local, free)
- Markdown-formatted responses
- Custom system prompt (default: technical teaching assistant)
- Persistent chat history within the session
- Stylish Gradio UI that renders inside Jupyter (`inline=True`)

---

## How to Run

### 1. Clone and navigate

```bash
cd 09_LLMs/AI_Technical_Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in this directory:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Get an OpenRouter key at: https://platform.openai.com/docs/quickstart

### 5. (Optional) Run Ollama locally

Download from https://ollama.com, then:

```bash
ollama run llama3
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `AI_Tech_Assistant.ipynb` and run all cells.

---

## Tech Stack

`Python` · `OpenAI API` · `Ollama` · `Gradio` · `Jupyter Notebook`
