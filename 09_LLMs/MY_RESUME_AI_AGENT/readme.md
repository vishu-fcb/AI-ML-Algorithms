# Resume AI Agent

> An agentic CV chatbot powered by OpenAI GPT with tool use, plus a background Gemini 2.0 Flash evaluator that monitors response quality without blocking the user.

---

## Overview

This project demonstrates two advanced LLM patterns:

1. **Tool-augmented agents** — the chatbot can call external tools (LinkedIn profile, speaker bio, lead capture) based on the conversation context
2. **Asynchronous LLM judge** — Gemini 2.0 Flash evaluates every response in the background and flags quality issues via Pushover push notifications, without slowing down the user-facing experience

---

## Features

- Acts as a CV assistant — answers questions about skills, experience, and career background
- **4 tools available to the agent:**
  - `record_user_details` — captures email/name for contact follow-up
  - `record_unknown_question` — logs questions the agent can't answer
  - `get_linkedin` — shares LinkedIn profile link
  - `get_speaker_profile` — shares conference speaker highlights
- **Background Gemini evaluator** — runs in a separate thread, assesses response quality without blocking
- Deployable locally or on Hugging Face Spaces
- Gradio web interface

---

## How to Run

### 1. Clone and navigate

```bash
cd 09_LLMs/MY_RESUME_AI_AGENT
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in this directory:

```
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-gemini-key
PUSHOVER_TOKEN=your-pushover-token      # optional — for push notifications
PUSHOVER_USER=your-pushover-user-key    # optional
```

### 5. Update the knowledge base

Replace the files in `me/` with your own profile:

| File | Purpose |
|---|---|
| `Knowledge_worker.md` | Your CV and professional background |
| `Overview.txt` | Short summary context |
| `Profile.pdf` | Additional profile document |

### 6. Run

```bash
python my_knowledge_worker.py
```

The Gradio interface launches in your browser.

---

## Tech Stack

`Python` · `OpenAI GPT` · `Google Gemini 2.0 Flash` · `Gradio` · `Pushover`
