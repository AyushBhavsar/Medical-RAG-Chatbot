# Medical Chatbot - Quick Start Guide

## Project Overview
A Flask-based medical chatbot with RAG (Retrieval Augmented Generation) using:
- **Pinecone**: Vector database for medical knowledge
- **HuggingFace**: flan-t5-large model for Q&A
- **LangChain**: Document processing and retrieval

## Project Structure
```
medical-chatbot/
├── app.py                    # Flask application (UPDATED)
├── src/
│   ├── helper.py            # Helper functions (UPDATED)
│   └── prompt.py            # System prompts
├── tempalates/
│   └── chat.html            # Chat interface
├── static/
│   └── style.css            # Styling
├── data/                    # PDF medical documents
├── research/
│   └── trials.ipynb         # Development notebook
├── .env                     # Environment variables
└── requirements.txt         # Dependencies (UPDATED)
```

## Files Updated Based on trials.ipynb

### 1. **helper.py** - Updated imports and function names
- Changed to use `langchain_community` for loaders and embeddings
- Function name: `load_pdf_files()` (was `load_pdf_file()`)
- Uses `sentence-transformers/all-MiniLM-L6-v2` embeddings

### 2. **app.py** - Complete overhaul
- **Replaced**: requests-based HTTP calls → `InferenceClient`
- **Model**: flan-t5-small → `google/flan-t5-large`
- **Added**: Lazy RAG initialization (faster startup)
- **Improved**: Error handling and context retrieval

### 3. **prompt.py** - Already correct
- System prompt matches notebook implementation

### 4. **requirements.txt** - Added dependency
- Added: `huggingface-hub==0.28.1`

## Setup Instructions

### 1. Activate Virtual Environment
```bash
# Windows PowerShell
.\chat\Scripts\Activate.ps1

# Windows CMD
.\chat\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Environment Variables
Check `.env` file has:
```
PINECONE_API_KEY=pcsk_...
OPENAI_API_KEY=sk-proj-...  
HUGGINGFACEHUB_API_TOKEN=hf_...
```

### 4. Test Setup (Optional)
```bash
python test_app.py
```

### 5. Run Application
```bash
python app.py
```

The app will start at: **http://127.0.0.1:8080**

## Key Changes from trials.ipynb

### InferenceClient Implementation
**Before (requests):**
```python
response = requests.post(HF_API_URL, headers=headers, json=payload)
```

**After (InferenceClient):**
```python
client = InferenceClient(token=huggingface_token)
response = client.text_generation(
    prompt,
    model="google/flan-t5-large",
    max_new_tokens=250,
    temperature=0.7
)
```

### RAG Flow
1. **Load embeddings** - `sentence-transformers/all-MiniLM-L6-v2`
2. **Connect to Pinecone** - Index: `medical-chatbot-index`
3. **Retrieve context** - Top 3 similar documents (k=3)
4. **Generate answer** - Using flan-t5-large with context

### Error Handling
- Graceful fallback if Pinecone is unavailable
- Runs without RAG (direct LLM) if needed
- Comprehensive error messages

## Features

### ✅ Working Features
- PDF document loading and chunking
- Vector embeddings with HuggingFace
- Pinecone vector search
- RAG-based question answering
- Flask web interface
- Lazy initialization (faster startup)

### 🔄 How It Works
1. User asks a medical question
2. App retrieves relevant context from Pinecone
3. Combines context + question into prompt
4. Sends to flan-t5-large via InferenceClient
5. Returns concise medical answer

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Deprecation warnings
These are warnings only - app still works. The imports have been updated to the latest standards.

### Pinecone connection fails
- Check `PINECONE_API_KEY` in `.env`
- App will run without RAG (direct LLM mode)

### HuggingFace API errors
- Verify `HUGGINGFACEHUB_API_TOKEN` in `.env`
- Model may be loading (wait 30-60 seconds)

## Testing the Chat

Example questions to try:
- "What is Allergic rhinitis?"
- "What are the symptoms of diabetes?"
- "How is hypertension treated?"

The chatbot uses medical PDFs in the `data/` folder for context-aware answers.

## Production Deployment

For production, replace Flask's built-in server:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

---

**Status**: ✅ All files updated and ready to run!
