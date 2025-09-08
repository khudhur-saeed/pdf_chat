# ==============================================================================
# Simple Multi-User ChatBot with Memory - All in One File
# ==============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import uuid
import os
import dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.partition.pdf import partition_pdf

# Load environment variables
dotenv.load_dotenv()

# ==============================================================================
# Data Models
# ==============================================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int

# ==============================================================================
# Simple Memory Manager
# ==============================================================================

class SimpleMemory:
    def __init__(self, max_messages: int = 50):
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.max_messages = max_messages
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        message = ChatMessage(role=role, content=content)
        self.sessions[session_id].append(message)
        
        # Keep only recent messages
        if len(self.sessions[session_id]) > self.max_messages:
            self.sessions[session_id] = self.sessions[session_id][-self.max_messages:]
    
    def get_history(self, session_id: str) -> str:
        """Get conversation history as formatted string"""
        if session_id not in self.sessions:
            return ""
        
        history = []
        for msg in self.sessions[session_id][-10:]:  # Last 6 messages for context
            history.append(f"{msg.role.upper()}: {msg.content}")
        
        return "\n".join(history)
    
    def get_message_count(self, session_id: str) -> int:
        """Get total message count for session"""
        return len(self.sessions.get(session_id, []))
    
    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]

# ==============================================================================
# Simple ChatBot
# ==============================================================================

class SimpleChatBot:
    def __init__(self):
        # Resolve API key (support multiple env names)
        api_key = os.getenv("LLM_KEY") or os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing LLM_KEY/GEMINI_KEY/GOOGLE_API_KEY in environment or .env")

        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=api_key
        )

        # Initialize embeddings and vector store (prefer Ollama, fallback to Google)
        try:
            self.embedding = OllamaEmbeddings(model="nomic-embed-text")
            # Quick probe to ensure Ollama is reachable
            _ = self.embedding.embed_query("ping")
            print("✅ Using Ollama embeddings (nomic-embed-text)")
        except Exception as e:
            print(f"⚠️ Ollama embeddings unavailable, falling back to Google: {e}")
            self.embedding = GoogleGenerativeAIEmbeddings(
                model="text-embedding-004",
                google_api_key=api_key
            )
            print("✅ Using Google Generative AI embeddings (text-embedding-004)")
        self.vectordb = None
        
        # Initialize memory
        self.memory = SimpleMemory(max_messages=10)
        
        # Response template
        self.template = """
        أنت مساعد ذكي وودود. تتحدث بشكل طبيعي مع المستخدمين.

        قواعد اللغة:
        - إذا كتب المستخدم بالعربية: رد بالعربية العراقية الطبيعية
        - إذا كتب بالإنجليزية: رد بالإنجليزية
        - إذا كتب بالتركية: رد بالتركية

        أمثلة على اللهجة العراقية:
        - استخدم "شلونك؟" بدلاً من "كيف حالك؟"
        - استخدم "أكو" بدلاً من "يوجد"
        - استخدم "شنو" بدلاً من "ماذا"
        - استخدم "وين" بدلاً من "أين"
        - استخدم "زين" بدلاً من "جيد"

        المحادثة السابقة:
        {history}

        معلومات من الوثائق:
        {context}

        سؤال المستخدم: {question}

        الرد:"""
    
    def load_pdf(self, pdf_path: str, persist_dir: str = "vector_db", collection: str = "docs"):
        """Load PDF and create vector database"""
        try:
            # Validate path
            if not os.path.isfile(pdf_path):
                print(f"❌ Error loading PDF: file not found -> {pdf_path}")
                return False

            # Read PDF with a lightweight strategy to avoid system deps (tesseract/poppler)
            # For higher quality OCR on scanned PDFs, switch to strategy="hi_res" but you'll need:
            #   sudo apt-get install -y tesseract-ocr poppler-utils
            elements = partition_pdf(pdf_path, strategy="fast", languages=["ar", "en"])
            texts = "\n".join([str(element.text) for element in elements])
            
            # Split text
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_text(texts)
            
            # Create vector database
            self.vectordb = Chroma.from_texts(
                texts=chunks,
                embedding=self.embedding,
                persist_directory=persist_dir,
                collection_name=collection
            )
            print(f"✅ Loaded PDF: {len(chunks)} chunks created")
            return True
            
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return False
    
    def load_existing_db(self, persist_dir: str = "vector_db", collection: str = "docs"):
        """Load existing vector database"""
        try:
            self.vectordb = Chroma(
                persist_directory=persist_dir,
                collection_name=collection,
                embedding_function=self.embedding
            )
            print("✅ Loaded existing vector database")
            return True
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False
    
    def chat(self, session_id: str, user_message: str) -> str:
        """Main chat function"""
        try:
            # Add user message to memory
            self.memory.add_message(session_id, "user", user_message)
            
            # Get conversation history
            history = self.memory.get_history(session_id)
            
            # Get relevant documents
            context = ""
            if self.vectordb:
                try:
                    docs = self.vectordb.similarity_search(user_message, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                except:
                    context = "لا توجد معلومات متاحة في الوثائق"
            
            # Create prompt
            prompt = ChatPromptTemplate.from_template(self.template)
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "history": history,
                "context": context,
                "question": user_message
            })
            
            # Add response to memory
            self.memory.add_message(session_id, "assistant", response)
            
            return response
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return "عذراً، حدث خطأ. جرب مرة أخرى."

# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(title="Simple Multi-User ChatBot")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = SimpleChatBot()

@app.on_event("startup")
async def startup():
    """Load vector database on startup"""
    # Try to load existing database first
    if not chatbot.load_existing_db():
        print("No existing database found. You need to load a PDF first.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint"""
    response = chatbot.chat(request.session_id, request.message)
    message_count = chatbot.memory.get_message_count(request.session_id)
    
    return ChatResponse(
        response=response,
        session_id=request.session_id,
        message_count=message_count
    )

@app.post("/create-session")
async def create_session():
    """Create new session"""
    return {"session_id": str(uuid.uuid4())}

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get session history"""
    if session_id not in chatbot.memory.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = chatbot.memory.sessions[session_id]
    return {
        "session_id": session_id,
        "messages": [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp} 
                    for msg in messages]
    }

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session"""
    chatbot.memory.clear_session(session_id)
    return {"message": "Session cleared"}

@app.post("/load-pdf")
async def load_pdf(pdf_path: str):
    """Load PDF file"""
    success = chatbot.load_pdf(pdf_path)
    if success:
        return {"message": "PDF loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to load PDF")

@app.get("/stats")
async def get_stats():
    """Get system stats"""
    return {
        "total_sessions": len(chatbot.memory.sessions),
        "total_messages": sum(len(msgs) for msgs in chatbot.memory.sessions.values())
    }

# ==============================================================================
# Run Server
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Simple Multi-User ChatBot")
    print("📖 API Docs: http://localhost:8000/docs")
    print("💡 Don't forget to load your PDF first!")
    
    uvicorn.run(
        "main:app",  # Change this to your filename
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# ==============================================================================
# Usage Examples
# ==============================================================================

"""
🔧 Setup:
1. pip install fastapi uvicorn langchain-google-genai langchain-ollama langchain-community chromadb unstructured python-dotenv
2. Create .env file: GEMINI_KEY=your_api_key
3. python main.py

📝 Usage Examples:

import requests

# 1. Create session
session = requests.post("http://localhost:8000/create-session").json()
session_id = session["session_id"]

# 2. Load PDF (first time only)
requests.post("http://localhost:8000/load-pdf", params={"pdf_path": "your_file.pdf"})

# 3. Chat
chat_data = {
    "message": "شلونك؟",
    "session_id": session_id
}
response = requests.post("http://localhost:8000/chat", json=chat_data)
print(response.json()["response"])

# 4. Another message (same session = remembers context)
chat_data["message"] = "شكراً لك"
response = requests.post("http://localhost:8000/chat", json=chat_data)
print(response.json()["response"])

# 5. Get history
history = requests.get(f"http://localhost:8000/session/{session_id}/history")
print(history.json())

# 6. Get stats
stats = requests.get("http://localhost:8000/stats")
print(stats.json())
"""