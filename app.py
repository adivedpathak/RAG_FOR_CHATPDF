from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from functools import lru_cache
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "docchat-index"
MAX_SESSIONS = 100
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500

# Session management
sessions: Dict[str, ChatMessageHistory] = {}
vectorstores: Dict[str, PineconeVectorStore] = {}

# Lazy-loaded model providers
@lru_cache()
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache()
def get_llm():
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

@lru_cache()
def get_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)

# Ensure Pinecone index exists
try:
    pc = get_pinecone()
    index_list = pc.list_indexes()
    existing_indexes = [index.name for index in index_list]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://doc-chat-ai-dgsp.vercel.app", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str]

class BytesIOPDFLoader:
    def __init__(self, file_content: BytesIO):
        self.file_content = file_content

    def load(self) -> List[Document]:
        reader = PdfReader(self.file_content)
        documents = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"page_number": i + 1, "total_pages": len(reader.pages)},
            )
            for i, page in enumerate(reader.pages)
            if page.extract_text()
        ]
        return documents

# Utility functions
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]

def get_vectorstore(session_id: str) -> PineconeVectorStore:
    if session_id not in vectorstores:
        vectorstores[session_id] = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=get_embeddings(),
            namespace=session_id
        )
    return vectorstores[session_id]

# API Routes
@app.get("/")
async def root():
    return {"message": "This is API of RAG model of project Chat with PDF by Aditya Vedpathak"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/start-session")
async def start_session():
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=400, detail="Max session limit reached")
    session_id = str(uuid4())
    sessions[session_id] = ChatMessageHistory()
    return {"session_id": session_id}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), session_id: str = Form(None)):
    if not session_id:
        session_id = str(uuid4())
        sessions[session_id] = ChatMessageHistory()

    try:
        all_documents = []
        for uploaded_file in files:
            content = await uploaded_file.read()
            loader = BytesIOPDFLoader(BytesIO(content))
            docs = loader.load()
            all_documents.extend(docs)

        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid documents were found.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(all_documents)

        vectorstores[session_id] = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            index_name=INDEX_NAME,
            namespace=session_id
        )

        return {"message": "Files uploaded and indexed successfully.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        retriever = get_vectorstore(request.session_id).as_retriever()

        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an intelligent assistant helping a retrieval system. Your task is to transform the latest user message into a fully self-contained, standalone question, using the full conversation history for clarity."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(get_llm(), retriever, contextual_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a top-tier assistant specialized in answering questions based strictly on provided context. Respond clearly, accurately, and concisely. If unsure, say: 'I'm not sure based on the available information.'\n\n=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_chain = create_stuff_documents_chain(get_llm(), qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        chat_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = chat_chain.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )

        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    if session_id in vectorstores:
        try:
            vectorstores[session_id].delete(delete_all=True)
        except:
            pc = get_pinecone()
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True, namespace=session_id)
        del vectorstores[session_id]
    return {"message": "Session cleaned up successfully."}

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
