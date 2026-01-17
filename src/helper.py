from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


def load_pdf_files(data):
    """
    Extract data from PDF files in the specified directory.
    
    Args:
        data: Path to directory containing PDF files
        
    Returns:
        List of Document objects
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to contain only 'source' in metadata and original page_content.
    
    Args:
        docs: List of Document objects
        
    Returns:
        List of filtered Document objects
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_split(filtered_data):
    """
    Split documents into smaller chunks for embedding.
    
    Args:
        filtered_data: List of Document objects to split
        
    Returns:
        List of text chunks as Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    texts_chunk = text_splitter.split_documents(filtered_data)
    return texts_chunk


def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings object
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

