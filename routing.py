from fastapi import FastAPI
import uvicorn
import main

main = main.main()
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello how can i help you?"}

app.post("/chat")
async def get_message(message:str):
    response = main.user_query(message)
    return response