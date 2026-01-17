from flask import Flask, render_template, request, session
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from src.prompt import system_prompt

from groq import Groq
from collections import defaultdict, deque


# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.secret_key = "change-this-secret-key"  # required for session memory

pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = pinecone_api_key

print("✅ Environment variables loaded")


# ---------------------------------------------------
# Groq Client
# ---------------------------------------------------

client = Groq(api_key=groq_api_key)
print("✅ Groq client initialized")


# ---------------------------------------------------
# RAG Globals
# ---------------------------------------------------

embeddings = None
retriever = None
use_rag = False


def init_rag():
    global embeddings, retriever, use_rag

    if retriever:
        return

    print("🔄 Initializing RAG...")

    embeddings = download_embeddings()

    docsearch = PineconeVectorStore(
        embedding=embeddings,
        index_name="medical-chatbot-index"
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    use_rag = True
    print("✅ RAG initialized successfully")


# ---------------------------------------------------
# Conversation Memory (LAST 10 MESSAGES PER USER)
# ---------------------------------------------------

conversation_memory = defaultdict(lambda: deque(maxlen=10))


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg", "").strip()
    print(f"👤 User Input: {user_input}")

    if not user_input:
        return "Please enter a valid medical question."

    # ---------------------------------------------------
    # Session handling
    # ---------------------------------------------------

    if "session_id" not in session:
        session["session_id"] = os.urandom(16).hex()

    session_id = session["session_id"]

    # Save user message
    conversation_memory[session_id].append(f"User: {user_input}")

    # ---------------------------------------------------
    # Initialize RAG if needed
    # ---------------------------------------------------

    if not use_rag:
        init_rag()

    # ---------------------------------------------------
    # Retrieve context from Pinecone
    # ---------------------------------------------------

    context = ""
    if retriever:
        docs = retriever.invoke(user_input)
        context = "\n\n".join(d.page_content for d in docs[:3])
        print(f"📚 Retrieved {len(docs)} documents")

    # ---------------------------------------------------
    # Build conversation history
    # ---------------------------------------------------

    chat_history = "\n".join(conversation_memory[session_id])

    # ---------------------------------------------------
    # Final Prompt (Memory + RAG)
    # ---------------------------------------------------

    final_prompt = f"""
You are a medical assistant.

Conversation history:
{chat_history}

Relevant medical information:
{context}

User question:
{user_input}

Answer clearly and concisely.
Always state this is for educational purposes and not professional medical advice.
"""

    # ---------------------------------------------------
    # Groq Chat Completion
    # ---------------------------------------------------

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ ACTIVE GROQ MODEL
            messages=[
                {"role": "system", "content": final_prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Groq error:", e)
        answer = "The AI model is temporarily unavailable. Please try again."

    # Save assistant response
    conversation_memory[session_id].append(f"Assistant: {answer}")

    print(f"🤖 Response: {answer}")
    return answer


# ---------------------------------------------------
# Optional: Clear Conversation Memory
# ---------------------------------------------------

@app.route("/clear", methods=["POST"])
def clear_chat():
    if "session_id" in session:
        conversation_memory.pop(session["session_id"], None)
    return "Conversation cleared."


# ---------------------------------------------------
# Run App
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)
