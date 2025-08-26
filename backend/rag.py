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
    
    try:
        path = "../data/"  # Data directory is at project root level
        
        # Check if data directory exists
        if not os.path.exists(path):
            print(f"Warning: Data directory '{path}' not found. RAG system will not be available.")
            return
        
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("Warning: No PDF documents found in data directory. RAG system will not be available.")
            return
        
        # Improved chunking for textbooks and manuals - larger chunks to preserve context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_documents = text_splitter.split_documents(docs)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
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
You are a helpful assistant who answers questions based on provided context. You must only use the provided context to answer the question. If you do not know the answer, or it's not contained in the provided context response with "I don't know"

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
  return {"response" : response.content}

class State(TypedDict):
  question: str
  context: List[Document]
  response: str

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()