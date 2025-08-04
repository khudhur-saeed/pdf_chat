from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import main  # هذا هو ملفك الذي يحتوي على الدوال

app = FastAPI()


# ✅ كود السماح بالطلبات من أي مصدر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # مؤقتًا أثناء التطوير
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# تحميل قاعدة البيانات
vectordb = main.get_vectordb("arabic_stories", "story")

# نموذج البيانات الذي سيُرسل من المستخدم
class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Hello, how can I help you?"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = main.user_query(request.message, vectordb)
    return {"response": response}
