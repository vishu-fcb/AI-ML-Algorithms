# Vishal Mishra RAG Chatbot 🤖

This project is an **AI-powered assistant** built with **LangChain** and **Gradio**, designed to answer professional questions about **Vishal Mishra**.  
It uses **Retrieval-Augmented Generation (RAG)** with a knowledge base of Vishal’s CV and career profile, enabling fact-based, contextual responses.  

---

## 🚀 Features
- 📂 Loads Vishal’s profile from a **Markdown file** and FAQ entries.
- 🔍 Uses **Chroma vector store** + **OpenAI embeddings** for retrieval.
- 💬 Interactive chatbot with **conversation memory**.
- ⚡ Cached FAQs for faster responses to common questions.
- 🎨 User-friendly **Gradio web UI** with example questions.

---


## 📦 Installation

Clone the repository:

- git clone https://github.com/vishu-fcb/AI-ML-Algorithms.git
- cd <Browse to AI_Powered_Chatbot>

## Create a virtual environment (recommended):

- python -m venv venv
- source venv/bin/activate   # Mac/Linux
- venv\Scripts\activate      # Windows

## Install dependencies:
- pip install -r requirements.txt

## 🔑 Environment Setup

- This project uses OpenAI API for embeddings and chat.
- Create a .env file in the root directory:
  - OPENAI_API_KEY=sk-your-openai-key

## Running the App

- python app.py
- The Gradio app will launch in your browser.
- You’ll see a clean UI where you can chat with the assistant.

## 🧠 How It Works

Loads Markdown CV and splits it into chunks with LangChain’s RecursiveCharacterTextSplitter.

Embeds chunks into a ChromaDB vector store with OpenAIEmbeddings.

Adds custom FAQ entries for common queries.

Uses ConversationalRetrievalChain with ChatOpenAI to generate answers based on retrieved context.

Caches FAQ responses for speed.

Provides an interactive Gradio interface with preloaded suggested questions.

## Example Questions

What are Vishal Mishra’s skills?

What is Vishal’s current role?

Tell me something different about Vishal’s career experience.

Where can I see more info about Vishal Mishra?

##  Notes

This assistant only answers questions about Vishal Mishra.

For unrelated questions, it will politely decline.

If sharing this repo publicly, consider using a sample CV/Markdown file instead of your real personal data.


