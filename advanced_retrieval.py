import os
from getpass import getpass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set Cohere API key from environment variable
if os.getenv("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.getenv("COHERE_API_KEY")
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import time
import asyncio
from ragas.testset import TestsetGenerator
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextEntityRecall, ContextPrecision
from ragas import evaluate, RunConfig


os.environ["LANGCHAIN_TRACING_V2"] = "false" 

def enable_tracing_for_evaluation():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"MusicMateAI_RAG_Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def disable_tracing():
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Global variables for lazy initialization
docs = None
split_documents = None
embeddings = None
client = None
vectorstore = None
retriever = None

def initialize_advanced_rag():
    """Initialize advanced RAG system lazily when needed"""
    global docs, split_documents, embeddings, client, vectorstore, retriever
    
    if retriever is not None:
        return  # Already initialized
    
    try:
        # Use the same path as rag.py
        path = "data/"
        
        # Check if data directory exists
        if not os.path.exists(path):
            print(f"Warning: Data directory '{path}' not found. Advanced RAG system will not be available.")
            return
        
        loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("Warning: No PDF documents found in data directory. Advanced RAG system will not be available.")
            return
        
        # Use the same chunk size as rag.py for consistency
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_documents = text_splitter.split_documents(docs)
        
        # Only create embeddings when needed
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        client = QdrantClient(":memory:")
        
        client.create_collection(
            collection_name="MusicMateAI_Contextual_Compression_Data",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="MusicMateAI_Contextual_Compression_Data",
            embedding=embeddings,
        )
        
        vectorstore.add_documents(split_documents)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k" : 4})
        
        print(f"Advanced RAG system initialized successfully with {len(split_documents)} document chunks")
        
    except Exception as e:
        print(f"Error initializing advanced RAG system: {e}")
        print("Advanced RAG system will not be available.")

def retrieve_contextual(state):
    initialize_advanced_rag()  # Initialize RAG system if needed
    if retriever is None:
        return {"context": []}  # Return empty context if RAG not available
    
    try:
        # Try contextual compression with Cohere
        compressor = CohereRerank(model="rerank-v3.5", top_n=10)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=retriever, 
            search_kwargs={"k": 5}
        )
        retrieved_docs = compression_retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"Cohere rerank failed: {e}")
        print("Falling back to basic retrieval...")
        # Fallback to basic retrieval
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}


# Global variables for parent document retriever
parent_document_retriever = None

def initialize_parent_document_retriever():
    """Initialize parent document retriever lazily when needed"""
    global parent_document_retriever, docs
    
    if parent_document_retriever is not None:
        return  # Already initialized
    
    try:
        # Initialize the main RAG system first to get docs
        initialize_advanced_rag()
        if docs is None:
            print("Warning: Main RAG system not available. Parent document retriever will not be available.")
            return
        
        parent_docs = docs
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=1500)
        
        client = QdrantClient(location=":memory:")
        
        client.create_collection(
            collection_name="MusicMateAI_Parent_Document_Data",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        
        parent_document_vectorstore = QdrantVectorStore(
            collection_name="MusicMateAI_Parent_Document_Data", embedding=OpenAIEmbeddings(model="text-embedding-3-small"), client=client
        )
        
        store = InMemoryStore()
        
        parent_document_retriever = ParentDocumentRetriever(
            vectorstore = parent_document_vectorstore,
            docstore=store,
            child_splitter=child_splitter,
        )
        
        parent_document_retriever.add_documents(parent_docs, ids=None)
        
        print("Parent document retriever initialized successfully")
        
    except Exception as e:
        print(f"Error initializing parent document retriever: {e}")
        print("Parent document retriever will not be available.")

def retrieve_parent_documents(state):
    initialize_parent_document_retriever()  # Initialize parent document retriever if needed
    if parent_document_retriever is None:
        return {"context": []}  # Return empty context if retriever not available
    
    try:
        retrieved_docs = parent_document_retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"Error in retrieve_parent_documents: {e}")
        return {"context": []}


RAG_PROMPT = """\
You are a helpful music production assistant who answers questions based ONLY on the provided context about music theory, production techniques, and DAW information. 
IMPORTANT: You must NOT use any external knowledge. If the context doesn't contain the answer, say "I don't have enough information in the provided documents to answer this question."

Context:
{context}

Question:
{question}

Answer (use ONLY information from the context above):
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)


def generate(state):
  if not state["context"]:
    return {"response": "I don't have enough information to answer that question. Please try rephrasing or ask about a different topic."}
  
  # Create model only when needed - using gpt-4o for better rate limits than mini
  openai_chat_model = ChatOpenAI(model="gpt-4o")
  docs_content = "\n\n".join(doc.page_content for doc in state["context"])
  messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
  response = openai_chat_model.invoke(messages)
  return {"response" : response.content}


class State(TypedDict):
  question: str
  context: List[Document]
  response: str

contextual_compression_graph_builder = StateGraph(State).add_sequence([retrieve_contextual, generate])
contextual_compression_graph_builder.add_edge(START, "retrieve_contextual")
contextual_compression_graph = contextual_compression_graph_builder.compile()

parent_document_graph_builder = StateGraph(State).add_sequence([retrieve_parent_documents, generate])
parent_document_graph_builder.add_edge(START, "retrieve_parent_documents")
parent_document_graph = parent_document_graph_builder.compile()

# Tool belt removed - not needed for RAG evaluation
# This file focuses on testing advanced retrieval techniques

# Initialize generator components lazily
generator_llm = None
generator_embeddings = None
generator = None

def initialize_generator():
    """Initialize the test generator lazily"""
    global generator_llm, generator_embeddings, generator
    
    if generator is not None:
        return  # Already initialized
    
    try:
        generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
        generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
        print("Test generator initialized successfully")
    except Exception as e:
        print(f"Error initializing test generator: {e}")
        print("Test generation will not be available.")


# Add delays between operations
print("Starting generation...")
initialize_advanced_rag()  # Initialize to get docs
initialize_generator()  # Initialize generator

# Test generation with improved rate limiting protection
if docs is not None and generator is not None:
    try:
        print("Generating test dataset with improved rate limiting protection...")
        print(f"Processing {len(docs)} documents...")
        
        # Process all documents but with better rate limiting
        dataset = generator.generate_with_langchain_docs(docs, testset_size=3)
        print("Generation complete!")
        
    except Exception as e:
        print(f"Error during test generation: {e}")
        if "rate limit" in str(e).lower() or "429" in str(e):
            print("Rate limit hit. Trying with smaller batch...")
            try:
                # Try with a much smaller subset
                print("Retrying with smaller document subset...")
                subset_docs = docs[:10]  # Use only first 10 documents
                dataset = generator.generate_with_langchain_docs(subset_docs, testset_size=3)
                print("Generation complete with smaller subset!")
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                print("Trying with minimal subset...")
                try:
                    # Try with minimal subset
                    subset_docs = docs[:3]  # Use only first 3 documents
                    dataset = generator.generate_with_langchain_docs(subset_docs, testset_size=3)
                    print("Generation complete with minimal subset!")
                except Exception as e3:
                    print(f"All attempts failed. Rate limit suggestions:")
                    print("1. Wait for rate limits to reset (usually 1 hour)")
                    print("2. Run during off-peak hours")
                    print("3. Consider upgrading your OpenAI plan")
                    dataset = None
        else:
            print(f"Unknown error: {e}")
            dataset = None
else:
    print("No documents or generator available for evaluation")
    dataset = None

# Enable tracing only for evaluation
enable_tracing_for_evaluation()
# Run your evaluation
# Then disable tracing
disable_tracing()

if dataset is not None:
    dataset.to_pandas()

    for test_row in dataset:
      response = contextual_compression_graph.invoke({"question" : test_row.eval_sample.user_input})
      test_row.eval_sample.response = response["response"]
      test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]


    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())

    # Create evaluator LLM only when needed
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", max_tokens=8192))

    custom_run_config = RunConfig(timeout=500)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[ContextPrecision(), Faithfulness(), ResponseRelevancy(), ContextEntityRecall()],
        llm=evaluator_llm,
        run_config=custom_run_config
    )
    result


    parent_evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())

    # Create parent evaluator LLM only when needed
    parent_evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", max_tokens=8192))

    custom_run_config = RunConfig(timeout=500)

    result = evaluate(
        dataset=parent_evaluation_dataset,
        metrics=[ContextPrecision(), Faithfulness(), ResponseRelevancy(), ContextEntityRecall()],
        llm=parent_evaluator_llm,
        run_config=custom_run_config
    )
    result
else:
    print("Skipping evaluation - no dataset available")
