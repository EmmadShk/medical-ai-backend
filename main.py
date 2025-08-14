from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import pdfplumber

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI medical assistant. Help patients discuss, understand, and identify health conditions, symptoms, diagnosis, and treatment options. Only provide educational advice, and recommend seeing a doctor for urgent or serious issues."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.3,
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        ai_response = f"Sorry, there was an error contacting the AI service: {e}"
    return {"response": ai_response}


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    file_contents = await file.read()
    text = ""
    # Basic file type handling
    if file.filename.lower().endswith(".pdf"):
        with open("temp.pdf", "wb") as f:
            f.write(file_contents)
        with pdfplumber.open("temp.pdf") as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        os.remove("temp.pdf")
    elif file.filename.lower().endswith(".txt"):
        text = file_contents.decode("utf-8")
    elif file.filename.lower().endswith(".csv"):
        text = file_contents.decode("utf-8")
    else:
        return {"analysis": "Unsupported file type. Please upload PDF, TXT, or CSV."}

    # Analyze with OpenAI
    try:
        prompt = (
            "You are a medical AI assistant. Analyze the following test results, "
            "explain what the findings mean in simple terms, and suggest what the patient should do next. "
            "If anything looks concerning, recommend seeing a doctor. Here are the results:\n\n" + text
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical AI assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3,
        )
        analysis = response.choices[0].message.content
    except Exception as e:
        analysis = f"Sorry, there was an error contacting the AI service: {e}"

    return {"analysis": analysis}