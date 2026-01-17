from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv(override=True)
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token

print(f"✅ Pinecone API Key loaded: {pinecone_api_key[:20]}...")
print(f"✅ OpenAI API Key loaded: {openai_api_key[:20]}...")
print(f"✅ HuggingFace Token loaded: {huggingface_token[:20]}...")

# Step 1: Load PDF documents
print("\n📄 Loading PDF documents...")
extracted_data = load_pdf_files("data")
print(f"✅ Loaded {len(extracted_data)} pages")

# Step 2: Filter documents
print("\n🔍 Filtering document metadata...")
filtered_data = filter_to_minimal_docs(extracted_data)
print(f"✅ Filtered {len(filtered_data)} documents")

# Step 3: Split text into chunks
print("\n✂️ Splitting text into chunks...")
texts_chunk = text_split(filtered_data)
print(f"✅ Created {len(texts_chunk)} text chunks")

# Step 4: Download embeddings model
print("\n🤖 Loading embeddings model...")
embeddings = download_embeddings()
print("✅ Embeddings model loaded")

# Step 5: Initialize Pinecone
print("\n📍 Initializing Pinecone...")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot-index"

# Step 6: Create index if it doesn't exist
if not pc.has_index(index_name):
    print(f"Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created")
else:
    print(f"✅ Index '{index_name}' already exists")

# Step 7: Upload vectors to Pinecone
print(f"\n⬆️ Uploading vectors to Pinecone (this may take a while)...")
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name
)

print("\n🎉 Vectors uploaded to Pinecone successfully!")
print(f"✅ Total chunks uploaded: {len(texts_chunk)}")
print("✅ Vector store is ready for use!")



