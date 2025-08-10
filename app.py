from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import uuid4
from io import BytesIO
import os
from functools import lru_cache

# Efficient PDF reader
import fitz  # PyMuPDF

# LangChain components
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# In-memory vector store (FAISS)
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_SESSIONS = 100
CHUNK_SIZE = 1024  # Adjusted for better performance with local models
CHUNK_OVERLAP = 200

# --- In-Memory Session Management ---
# sessions holds the chat history
sessions: Dict[str, ChatMessageHistory] = {}
# vectorstores will now hold the in-memory FAISS index
vectorstores: Dict[str, FAISS] = {}

# --- Lazy-loaded Models (Efficient Caching) ---
@lru_cache()
def get_embeddings():
    """Initializes the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@lru_cache()
def get_llm():
    """Initializes the Language Model."""
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# --- FastAPI App Initialization ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://doc-chat-ai-dgsp.vercel.app", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

# --- Efficient PDF Loader using PyMuPDF ---
class PyMuPDFLoader:
    """A highly efficient PDF loader that uses PyMuPDF (fitz)."""
    def __init__(self, file_content: bytes):
        self.file_content = file_content

    def load(self) -> List[Document]:
        documents = []
        with fitz.open(stream=self.file_content, filetype="pdf") as doc:
            for i, page in enumerate(doc):
                text = page.get_text()
                if text:  # Only add pages with actual text content
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"page_number": i + 1, "total_pages": len(doc)},
                        )
                    )
        return documents

# --- Utility Functions ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieves or creates a chat history for a given session."""
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]

# --- API Routes ---

@app.get("/")
async def root():
    return {"message": "API for Chat with PDF RAG model by Aditya Vedpathak"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/start-session")
async def start_session():
    """Starts a new session and returns a unique session ID."""
    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(status_code=503, detail="Max session limit reached")
    session_id = str(uuid4())
    sessions[session_id] = ChatMessageHistory()
    return {"session_id": session_id}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    """Uploads PDFs, creates embeddings, and stores them in an in-memory FAISS index."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please start a session first.")
    
    try:
        all_documents = []
        for uploaded_file in files:
            content = await uploaded_file.read()
            loader = PyMuPDFLoader(content)
            docs = loader.load()
            all_documents.extend(docs)

        if not all_documents:
            raise HTTPException(status_code=400, detail="No text could be extracted from the provided PDF(s).")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(all_documents)

        # Create FAISS index from documents and store it in memory
        vectorstores[session_id] = FAISS.from_documents(chunks, get_embeddings())

        return {"message": "Files uploaded and indexed successfully.", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    """Handles the main chat logic using the session's vector store."""
    if not request.session_id or request.session_id not in vectorstores:
        raise HTTPException(status_code=404, detail="Invalid session_id. Please upload documents first.")

    try:
        retriever = vectorstores[request.session_id].as_retriever()
        llm = get_llm()

        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answering prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # Add history to the chain
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
    """Clears a session's history and its vector store from memory."""
    if session_id in sessions:
        del sessions[session_id]
    if session_id in vectorstores:
        del vectorstores[session_id]
    return {"message": "Session cleaned up successfully."}

# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting FastAPI server on port {port}...")
    uvicorn.run("your_filename:app", host="0.0.0.0", port=port, reload=True)
    # Replace "your_filename" with the actual name of your python file
