"""Retrieval-Augmented Generation (RAG) utilities for Music Production Assistant.

This module builds an in-memory RAG pipeline that:
- helps the user find relevant information about music theory, production, and DAWs
- uses a vector database to store and retrieve information
- uses a language model to generate responses
- uses a graph to orchestrate the flow of the conversation

"""
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for lazy initialization
docs = None
split_documents = None
embeddings = None
client = None
vectorstore = None
retriever = None

def initialize_rag():
    """Initialize RAG system lazily when needed"""
    global docs, split_documents, embeddings, client, vectorstore, retriever
    
    if retriever is not None:
        return  # Already initialized
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables. RAG system will not be available.")
        print("Please set your OpenAI API key in the environment variables.")
        return
    
    try:
        # Use absolute path to data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "..", "data")
        
        # Check if data directory exists
        if not os.path.exists(path):
            print(f"Warning: Data directory '{path}' not found. RAG system will not be available.")
            return
        
        print(f"Loading documents from: {path}")
        
        # Load documents with error handling for corrupted PDFs
        docs = []
        pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(path, pdf_file)
                loader = PyMuPDFLoader(file_path)
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"Successfully loaded {len(file_docs)} pages from {pdf_file}")
            except Exception as e:
                print(f"Warning: Failed to load {pdf_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(docs)} total document pages")
        
        if not docs:
            print("Warning: No PDF documents could be loaded from data directory. RAG system will not be available.")
            return
        
        # Improved chunking for textbooks and manuals - larger chunks to preserve context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_documents = text_splitter.split_documents(docs)
        
        print(f"Split documents into {len(split_documents)} chunks")
        
        print("Initializing embeddings...")
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"Error initializing OpenAI embeddings: {e}")
            print("Please ensure OPENAI_API_KEY is set in your environment variables")
            return
        
        print("Creating vector store...")
        client = QdrantClient(":memory:")
        
        client.create_collection(
            collection_name="Music_Production_Data",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="Music_Production_Data",
            embedding=embeddings,
        )
        
        print("Adding documents to vector store...")
        vectorstore.add_documents(split_documents)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k" : 4})  # Reduced for faster retrieval
        
        print(f"RAG system initialized successfully with {len(split_documents)} document chunks")
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("RAG system will not be available.")

def retrieve(state):
  initialize_rag()  # Initialize RAG system if needed
  if retriever is None:
    return {"context": []}  # Return empty context if RAG not available
  retrieved_docs = retriever.invoke(state["question"])
  return {"context" : retrieved_docs}

RAG_PROMPT = """\
You are a helpful music production assistant. Answer the user's question based on the provided context from music production manuals and guides. 

Use the context to provide accurate, helpful information. If the context contains relevant information, use it to answer the question. If the context doesn't contain enough information to fully answer the question, provide what you can from the context and mention that you may need more specific information.

Keep your response concise and focused (aim for 2-4 sentences maximum).

Context:
{context}

Question:
{question}

Answer:
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

def generate(state):
  initialize_rag()  # Initialize RAG system if needed
  if not state["context"]:
    return {"response": "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."}
  
  # Create model only when needed
  openai_chat_model = ChatOpenAI(model="gpt-4o-mini")
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = openai_chat_model.invoke(messages)
  
  # Limit response length and clean it up
  response_text = response.content.strip()
  if len(response_text) > 500:  # Limit to 500 characters
    response_text = response_text[:497] + "..."
  
  return {"response": response_text}

class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def test_rag_initialization():
    """Test function to debug RAG initialization"""
    try:
        print("Testing RAG initialization...")
        initialize_rag()
        if retriever is not None:
            print("RAG initialization successful!")
            return True
        else:
            print("RAG initialization failed - retriever is None")
            return False
    except Exception as e:
        print(f"RAG initialization error: {e}")
        return False

if __name__ == "__main__":
    # Test the RAG system
    test_rag_initialization()