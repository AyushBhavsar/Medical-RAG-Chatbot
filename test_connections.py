"""
Test script to verify all connections and components
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

print("=" * 60)
print("TESTING MEDICAL CHATBOT CONNECTIONS")
print("=" * 60)

# 1. Test Environment Variables
print("\n1. Checking Environment Variables...")
pinecone_key = os.getenv("PINECONE_API_KEY")
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if pinecone_key:
    print(f"✓ PINECONE_API_KEY found: {pinecone_key[:20]}...")
else:
    print("✗ PINECONE_API_KEY not found!")

if huggingface_token:
    print(f"✓ HUGGINGFACEHUB_API_TOKEN found: {huggingface_token[:20]}...")
else:
    print("✗ HUGGINGFACEHUB_API_TOKEN not found!")

# 2. Test Embeddings
print("\n2. Testing Embeddings...")
try:
    from src.helper import download_embeddings
    embeddings = download_embeddings()
    print("✓ Embeddings loaded successfully")
    
    # Test embedding a sample text
    test_text = "test medical query"
    test_embedding = embeddings.embed_query(test_text)
    print(f"✓ Sample embedding generated (dimension: {len(test_embedding)})")
except Exception as e:
    print(f"✗ Embeddings error: {str(e)}")

# 3. Test Pinecone Connection
print("\n3. Testing Pinecone Connection...")
try:
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=pinecone_key)
    indexes = pc.list_indexes()
    print(f"✓ Connected to Pinecone")
    print(f"  Available indexes: {[idx.name for idx in indexes]}")
    
    # Check if medical-chatbot-index exists
    index_name = "medical-chatbot-index"
    index_names = [idx.name for idx in indexes]
    if index_name in index_names:
        print(f"✓ Index '{index_name}' exists")
        
        # Get index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"  Index stats: {stats}")
    else:
        print(f"✗ Index '{index_name}' NOT FOUND!")
        print(f"  You need to create the index first by running store.py")
        
except Exception as e:
    print(f"✗ Pinecone error: {str(e)}")
    import traceback
    traceback.print_exc()

# 4. Test HuggingFace LLM
print("\n4. Testing HuggingFace LLM...")
try:
    from langchain_community.llms import HuggingFaceHub
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=huggingface_token,
        task="text2text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 512
        }
    )
    print("✓ LLM initialized successfully")
    
    # Test with a simple query
    print("  Testing LLM with sample query...")
    response = llm.invoke("What is 2+2?")
    print(f"✓ LLM response: {response}")
    
except Exception as e:
    print(f"✗ LLM error: {str(e)}")
    import traceback
    traceback.print_exc()

# 5. Test Pinecone Vector Store
print("\n5. Testing Pinecone Vector Store...")
try:
    from langchain_pinecone import PineconeVectorStore
    
    docsearch = PineconeVectorStore(
        embedding=embeddings,
        index_name="medical-chatbot-index"
    )
    print("✓ Vector store connected")
    
    # Test retrieval
    print("  Testing similarity search...")
    results = docsearch.similarity_search("heart disease", k=2)
    print(f"✓ Retrieved {len(results)} documents")
    if results:
        print(f"  Sample result: {results[0].page_content[:100]}...")
    
except Exception as e:
    print(f"✗ Vector store error: {str(e)}")
    import traceback
    traceback.print_exc()

# 6. Test Complete RAG Chain
print("\n6. Testing Complete RAG Chain...")
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    prompt_text = (
        "You are a medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_text)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("✓ RAG chain created successfully")
    
    # Test with a medical question
    print("  Testing RAG chain with medical question...")
    test_question = "What is diabetes?"
    response = rag_chain.invoke({"input": test_question})
    answer = response.get("answer", "No answer")
    print(f"✓ RAG chain response: {answer}")
    
except Exception as e:
    print(f"✗ RAG chain error: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
