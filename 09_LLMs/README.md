# LLM Applications

> Three production-style AI applications built with Large Language Models — a RAG chatbot, a technical assistant, and an AI-powered CV agent.

---

## Overview

This folder contains real-world LLM application projects using OpenAI GPT models, Google Gemini, LangChain, and Gradio. Each project demonstrates a distinct LLM pattern:

| Pattern | Project |
|---|---|
| RAG (Retrieval-Augmented Generation) | AI_Powered_Chatbot |
| Dual-model chat (cloud + local) | AI_Technical_Assistant |
| Agentic AI with tools + evaluation loop | MY_RESUME_AI_AGENT |

---

## Projects

### [AI_Powered_Chatbot](AI_Powered_Chatbot/)

A RAG-based chatbot that answers professional questions about Vishal Mishra by retrieving relevant context from a knowledge base (CV + profile documents) before generating responses.

**Stack:** LangChain · ChromaDB · OpenAI Embeddings · Gradio
**Key pattern:** Document ingestion → vector store → conversational retrieval chain

---

### [AI_Technical_Assistant](AI_Technical_Assistant/)

A dual-model technical Q&A chatbot that lets you switch between OpenAI GPT-4o-mini (cloud) and a locally-running Ollama LLaMA 3 model — all through a polished Gradio interface.

**Stack:** OpenAI API · Ollama (LLaMA 3) · Gradio · Jupyter Notebook
**Key pattern:** Model-agnostic chat interface with local/cloud toggle

---

### [MY_RESUME_AI_AGENT](MY_RESUME_AI_AGENT/)

An agentic CV assistant powered by OpenAI GPT with tool use (LinkedIn, speaker profile, lead capture) and a background Gemini 2.0 Flash evaluator that monitors response quality without slowing down the user experience.

**Stack:** OpenAI GPT · Google Gemini 2.0 Flash · Gradio · Pushover notifications
**Key pattern:** Tool-augmented agent + asynchronous LLM judge

---

## Prerequisites (All Projects)

```
Python 3.8+
OpenAI API key (required for all projects)
Google API key (MY_RESUME_AI_AGENT only)
Ollama installed locally (AI_Technical_Assistant only)
```

Each project has its own `requirements.txt` — see individual project READMEs for full setup instructions.

---

## API Keys

Each project reads keys from a `.env` file in its own directory:

```
OPENAI_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-gemini-key     # MY_RESUME_AI_AGENT only
```

Never commit `.env` files — they are gitignored.
