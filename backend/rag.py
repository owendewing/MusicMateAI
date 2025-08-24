# """Retrieval-Augmented Generation (RAG) utilities for Music Production Assistant.

# This module builds an in-memory RAG pipeline that:
# - Loads pdfs that are exported from files json_to_pdf.py, midi_to_pdf.py, and xml_to_pdf.py from structured_data folder
# - Splits documents into chunks using a token-aware splitter and overlap
# - Embeds chunks with OpenAI and stores vectors in an in-memory Qdrant store

# """
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict
# from langchain_core.documents import Document


# path = "../../unstructured_data/"
# loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# split_documents = text_splitter.split_documents(docs)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# client = QdrantClient(":memory:")

# client.create_collection(
#     collection_name="Music_Production_Data",
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
# )

# vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name="Music_Production_Data",
#     embedding=embeddings,
# )

# vectorstore.add_documents(split_documents)

# retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})

# def retrieve(state):
#   retrieved_docs = retriever.invoke(state["question"])
#   return {"context" : retrieved_docs}

# RAG_PROMPT = """\
# You are a helpful assistant who answers questions based on provided context. You must only use the provided context to answer the question. If you do not know the answer, or it's not contained in the provided context response with "I don't know"

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

# rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# openai_chat_model = ChatOpenAI(model="gpt-4.1-nano")

# def generate(state):
#   docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#   messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
#   response = openai_chat_model.invoke(messages)
#   return {"response" : response.content}

# class State(TypedDict):
#   question: str
#   context: List[Document]
#   response: str

# graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# graph_builder.add_edge(START, "retrieve")
# graph = graph_builder.compile()