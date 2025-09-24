# Vishal Mishra CV Chatbot 🤖

This is an **AI-powered chatbot** that represents Vishal Mishra’s CV and professional profile.  
It uses **OpenAI GPT** for responses and **Gemini 2.0 Flash** as a background evaluator to assess response quality.  

The chatbot runs in a Gradio interface and includes interactive tools such as:
- 📧 **Record user details** (lead capture via email)
- ❓ **Record unknown questions**
- 🔗 **Share LinkedIn profile**
- 🎤 **Show conference speaker profile**

---

## 🚀 Features
- Acts as a **CV chatbot** answering questions about Vishal’s background, skills, and experience.
- Integrates with **tools** for LinkedIn, conference talks, and lead capture.
- Runs a **Gemini evaluator** in the background to monitor response quality (without blocking user experience).
- Deployable locally or on cloud platforms like Hugging Face Spaces.

---

## 📦 Installation

Clone the repository:

git <clone the repo>
cd <navigate to the folder MY_RESUME_AI_AGENT>

# Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies 

pip install -r requirements.txt

# Create a .env file in the root directory and set:

OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-gemini-key
PUSHOVER_TOKEN=your-pushover-token
PUSHOVER_USER=your-pushover-user

# Run the app
python3 my_knowledge_worker.py

This will launch a Gradio web interface in your browser.
You can now chat with the CV assistant.

# Tools Available

record_user_details → Save email/name for contact
record_unknown_question → Capture unanswerable queries
get_linkedin → Share Vishal’s LinkedIn profile
get_speaker_profile → Show conference speaker highlights


