# AI-Powered RAG Chatbot

> A Retrieval-Augmented Generation (RAG) chatbot that answers professional questions about Vishal Mishra by grounding responses in a personal knowledge base.

---

## Overview

This project demonstrates the **RAG pattern** — one of the most important LLM application architectures. Instead of relying on the model's training data alone, the chatbot:

1. Loads a CV and profile documents into a **ChromaDB vector store**
2. At query time, **retrieves the most relevant chunks** using semantic search
3. Passes retrieved context to **ChatOpenAI** to generate a grounded, fact-based response

This prevents hallucination — the model only answers from documents you control.

---

## Architecture

```
Profile documents (Markdown + FAQs)
        │
        ▼
RecursiveCharacterTextSplitter
        │
        ▼
OpenAI Embeddings → ChromaDB vector store
        │
        ▼
User query → similarity search → top-K chunks
        │
        ▼
ConversationalRetrievalChain (ChatOpenAI)
        │
        ▼
Grounded response with conversation memory
```

---

## Features

- Loads profile from a Markdown file + custom FAQ entries
- ChromaDB vector store with OpenAI embeddings for semantic retrieval
- Conversation memory — the chatbot remembers earlier messages in the session
- Cached FAQ responses for faster answers to common questions
- Gradio web UI with preloaded example questions

---

## How to Run

### Prerequisites

```
Python 3.8+
OpenAI API key
```

### Install dependencies

```bash
cd 09_LLMs/AI_Powered_Chatbot
pip install -r requirements.txt
```

### Set up environment

Create a `.env` file in the `AI_Powered_Chatbot/` directory:

```
OPENAI_API_KEY=sk-your-openai-key-here
```

### Run

```bash
python chatbot_app.py
```

The Gradio app launches in your browser automatically.

---

## Knowledge Base

The chatbot reads from `My_knowledge_worker/`:

| File | Purpose |
|---|---|
| `Knowledge_worker.md` | Primary CV and profile content |
| `Overview.txt` | Summary context |
| `Profile.pdf` | Additional profile document |

To adapt this for your own profile, replace these files with your own content.

---

## Tech Stack

`Python` · `LangChain` · `ChromaDB` · `OpenAI` · `Gradio`
