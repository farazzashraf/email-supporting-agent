import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI(title="Developer's Agent")

# Define the expected JSON payload
class RunRequest(BaseModel):
    tenant_context: dict
    request: str

@app.post("/run")
async def run_agent(payload: RunRequest):
    # Setup Gemini with API key from environment
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set globally.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    business_name = payload.tenant_context.get("business_name", "the business")
    faq_text = payload.tenant_context.get("faq_text", "")
    message = payload.request

    prompt = f"""You are the AI assistant for {business_name}. Here is the business FAQ: {faq_text}. The user says: '{message}'. If they are explicitly trying to book an appointment, respond EXACTLY with this JSON: {{"action": "push_integration", "target": "booking_webhook"}}. Otherwise, answer them friendly using the FAQ and respond EXACTLY with this JSON: {{"reply": "Your answer here"}}."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        # Safely parse JSON even if the model wraps it in markdown blocks
        text_resp = response.text.strip()
        if text_resp.startswith("```json"):
            text_resp = text_resp[7:-3].strip()
        elif text_resp.startswith("```"):
            text_resp = text_resp[3:-3].strip()
            
        return json.loads(text_resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # The Developer's Agent runs on Port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
