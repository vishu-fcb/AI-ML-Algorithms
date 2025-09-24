import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import gradio as gr

# Load ENV and API key
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Load markdown file
folder_path = "My_knowledge_worker"
md_file_path = [f for f in os.listdir(folder_path) if f.endswith(".md")]
if not md_file_path:
    raise FileNotFoundError("No .md file found in the folder.")
full_path = os.path.join(folder_path, md_file_path[0])

loader = TextLoader(full_path, encoding='utf-8')
documents = loader.load()

# Assign section metadata based on content
for doc in documents:
    content = doc.page_content.lower()
    if "other skills" in content:
        doc.metadata["section"] = "OTHER SKILLS"
    elif "experience" in content:
        doc.metadata["section"] = "EXPERIENCE"
    elif "education" in content:
        doc.metadata["section"] = "EDUCATION"
    elif "skills" in content:
        doc.metadata["section"] = "SKILLS"
    elif "projects" in content:
        doc.metadata["section"] = "PROJECTS"
    else:
        doc.metadata["section"] = "general"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(documents)

faq_docs = [
    Document(
        page_content="Q: What is Vishal Mishra's LinkedIn?\nA: https://linkedin.com/in/your-link",
        metadata={"section": "faq"}
    ),
    Document(
        page_content="Q: What are Vishal's top skills?\nA: Automotive software developement, Programming, AUTOSAR, diagnostics, cybersecurity.",
        metadata={"section": "faq"}
    ),
    Document(
        page_content="Q: Where does Vishal work?\nA: Vishal works at Daimler Truck AG.",
        metadata={"section": "faq"}
    ),
    Document(
        page_content="Q: What is Vishal's current role?\nA: He works in Diagnostics and Secure Diagnostics on HPCs for SDVs.",
        metadata={"section": "faq"}
    )
]

chunks += faq_docs

# Recreate vectorstore
db_name = "knowledge_worker_db"
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=OpenAIEmbeddings()).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=db_name
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 14})
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
memory.chat_memory.messages.insert(0, SystemMessage(
    content="""You are an AI Assistant that provides professional, factual answers about Vishal Mishra. 
Only respond to questions explicitly related to Vishal. Do not make up information. 
If the user greets you, reply with: \"Hi! How can I help you today regarding Vishal Mishra?\"
Always suggest relevant follow-up questions like:
- Would you like to know his LinkedIn?
- Do you want to see his top skills or certifications?
- Interested in his recent work or projects?
If the question is unrelated, say:
\"I'm here to help only with questions about Vishal Mishra's background.\"
"""
))

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

faq_cache = {
    "What are Vishal Mishra's skills?": "Programming, Automotive software, AUTOSAR, Diagnostics & Cybersecurity, Software Defined Vehicles(SDVs)",
    "What is Vishal Mishra's present work profile?": "Vishal Mishra is currentl working at Daimler Truck AG as a Diagnostics and Cybersecurity architect.",
    "Tell me something different about Vishal Mishra's career experience?": "AUTOSAR trainer and a tech speaker at various Automotive conferences like ACC, Automotive IQ.",
    "Where I can see more info about Vishal Mishra? ": "Here is the link to his portfolio website: https://vishal-mishra-autoengine-r256d27.gamma.site/."
}

DEFAULT_SUGGESTIONS = list(faq_cache.keys())

# Build Gradio UI
with gr.Blocks(css="body {background-color: #f4f4f9;}") as demo:
    gr.Markdown("""
    <h1 style="text-align:center; color:#2c3e50;">🔎 Ask Me Anything About Vishal Mishra</h1>
    <p style="text-align:center; font-size:16px; color:#34495e;">
        This assistant is trained on Vishal Mishra's professional profile. Ask anything factual or explore different sections.
    </p>
    """)

    chatbot = gr.Chatbot()

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Ask something about Vishal Mishra...",
            show_copy_button=True,
            scale=7,
            lines=1
        )
        submit_btn = gr.Button("Send", scale=1)

    def respond(message, chat_history):
        normalized = message.strip().lower()
        if normalized in faq_cache:
            return "", chat_history + [[message, faq_cache[normalized]]]
        result = conversation_chain.invoke({"question": message})
        answer = result["answer"]
        chat_history.append([message, answer])
        return "", chat_history

    submit_btn.click(fn=respond, inputs=[txt, chatbot], outputs=[txt, chatbot])
    txt.submit(fn=respond, inputs=[txt, chatbot], outputs=[txt, chatbot])

    with gr.Accordion("Try These Questions", open=True):
        for q in DEFAULT_SUGGESTIONS:
            btn = gr.Button(q)
            btn.click(fn=respond, inputs=[gr.Textbox(value=q, visible=False), chatbot], outputs=[txt, chatbot])

    gr.Markdown("""
    <hr>
    <p style="text-align:center; font-size:14px; color:gray;">
    ⚠️ Note: This assistant only answers questions related to Vishal Mishra.
    </p>
    """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
