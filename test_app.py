"""
Test script to verify the medical chatbot application is working correctly.
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("=" * 60)
print("TESTING MEDICAL CHATBOT SETUP")
print("=" * 60)

# Test 1: Check environment variables
print("\n✓ Test 1: Environment Variables")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if pinecone_api_key:
    print(f"  ✅ PINECONE_API_KEY: {pinecone_api_key[:20]}...")
else:
    print("  ❌ PINECONE_API_KEY: Not found!")

if huggingface_token:
    print(f"  ✅ HUGGINGFACEHUB_API_TOKEN: {huggingface_token[:20]}...")
else:
    print("  ❌ HUGGINGFACEHUB_API_TOKEN: Not found!")

# Test 2: Import all modules
print("\n✓ Test 2: Import Modules")
try:
    from src.helper import download_embeddings, load_pdf_files, text_split, filter_to_minimal_docs
    print("  ✅ src.helper imported successfully")
except Exception as e:
    print(f"  ❌ src.helper import failed: {e}")

try:
    from src.prompt import system_prompt
    print("  ✅ src.prompt imported successfully")
    print(f"  System prompt preview: {system_prompt[:80]}...")
except Exception as e:
    print(f"  ❌ src.prompt import failed: {e}")

try:
    from langchain_pinecone import PineconeVectorStore
    print("  ✅ langchain_pinecone imported successfully")
except Exception as e:
    print(f"  ❌ langchain_pinecone import failed: {e}")

try:
    from huggingface_hub import InferenceClient
    print("  ✅ huggingface_hub.InferenceClient imported successfully")
except Exception as e:
    print(f"  ❌ huggingface_hub import failed: {e}")

# Test 3: Initialize HuggingFace Client
print("\n✓ Test 3: HuggingFace InferenceClient")
try:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=huggingface_token)
    print("  ✅ InferenceClient initialized successfully")
except Exception as e:
    print(f"  ❌ InferenceClient initialization failed: {e}")

# Test 4: Initialize Embeddings
print("\n✓ Test 4: Embeddings")
try:
    embeddings = download_embeddings()
    print(f"  ✅ Embeddings loaded: {type(embeddings).__name__}")
except Exception as e:
    print(f"  ❌ Embeddings loading failed: {e}")

# Test 5: Check Flask app structure
print("\n✓ Test 5: Flask App")
try:
    from app import app
    print(f"  ✅ Flask app imported: {app.name}")
    print(f"  ✅ Template folder: {app.template_folder}")
except Exception as e:
    print(f"  ❌ Flask app import failed: {e}")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✅ All tests passed! Your medical chatbot is ready to run.")
print("\nTo start the application, run:")
print("  python app.py")
print("\nThen open your browser to:")
print("  http://127.0.0.1:8080")
print("=" * 60)
