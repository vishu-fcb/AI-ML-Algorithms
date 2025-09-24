from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import threading

load_dotenv(override=True)

# --- Setup OpenAI (chatbot) ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Setup Gemini (evaluator, OpenAI-compatible API) ---
gemini = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"), 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

def get_linkedin():
    return {"url": "https://www.linkedin.com/in/vishal-mishra-300894"}  

def get_speaker_profile():
    return {
        "tagline": "Conference Speaker on Automotive Software & Diagnostics",
        "highlights": [
            "Invited Speaker – Automotive Computing Conference, Munich 2025",
            "Talks on AUTOSAR, Service-Oriented Vehicle Diagnostics (SOVD), and Software-Defined Vehicles",
            "Engages with audiences of engineers, researchers, and industry leaders"
        ],
        "contact": "Reach out via LinkedIn or email to book Vishal for conferences or panels."
    }


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string","description": "The email address of this user"},
            "name": {"type": "string","description": "The user's name, if they provided it"},
            "notes": {"type": "string","description": "Any additional information"},
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string","description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

get_linkedin_json = {
    "name": "get_linkedin",
    "description": "Provides a link to Vishal Mishra's LinkedIn profile so the user can connect with him.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

get_speaker_profile_json = {
    "name": "get_speaker_profile",
    "description": "Provides Vishal Mishra's conference speaker profile, including key highlights and booking info.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}


tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
    {"type": "function", "function": get_linkedin_json},
    {"type": "function", "function": get_speaker_profile_json}
]


# --- Evaluator setup ---
evaluator_system_prompt = """
You are an evaluator for a chatbot that represents Vishal Mishra's CV.
Your job is to evaluate the chatbot's response.
Return ONLY a JSON with this format:
{
  "score": 1-5,
  "verdict": "good" or "bad",
  "feedback": "short explanation"
}
"""

def evaluator_user_prompt(reply, message, history):
    return f"""
User message: "{message}"
Chatbot reply: "{reply}"
Conversation history: {history}

Evaluate on:
- Faithfulness to Vishal's CV
- Helpfulness
- Professionalism
"""

def run_evaluator_in_background(reply, message, history):
    """Run Gemini evaluator without blocking user response"""
    def worker():
        try:
            messages = [
                {"role": "system", "content": evaluator_system_prompt},
                {"role": "user", "content": evaluator_user_prompt(reply, message, history)}
            ]
            response = gemini.chat.completions.create(
                model="gemini-2.0-flash",
                messages=messages
            )
            try:
                evaluation = json.loads(response.choices[0].message.content)
            except:
                evaluation = {"score": 3, "verdict": "unknown", "feedback": "Evaluator failed to parse"}
            print("Evaluator:", evaluation, flush=True)
        except Exception as e:
            print("Evaluator error:", e, flush=True)

    threading.Thread(target=worker, daemon=True).start()


class Me:

    def __init__(self):
        self.openai = openai_client
        self.name = "Vishal Mishra"
        reader = PdfReader("me\Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me\Overview.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results
    
    def system_prompt(self):
        return f"""You are acting as {self.name}. 
You are answering questions on {self.name}'s website, 
particularly questions related to {self.name}'s career, background, skills and experience. 
Be professional and engaging. 
If you don't know the answer, use record_unknown_question. 
Encourage users to leave their email with record_user_details. 

## Summary:
{self.summary}

## LinkedIn Profile:
{self.linkedin}

Stay in character as {self.name}.
"""
    
    def generate_openai_answer(self, messages):
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools
            )
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content

    def chat(self, message, history):
        # Generate chatbot response immediately
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        ai_answer = self.generate_openai_answer(messages)

        # Start evaluator in background (non-blocking)
        run_evaluator_in_background(ai_answer, message, history)

        return ai_answer
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()