import os
import io
import json

import PyPDF2
import docx2txt
import openai

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="NexusAI Resume Parser")

# ───────────────────────────────────────────
# HARDCODED CREDENTIALS
# ───────────────────────────────────────────
API_KEY  = os.getenv("API_KEY")
BASE_URL = os.getenv("NAVIGATE_BASE_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────
# TEXT EXTRACTION
# ───────────────────────────────────────────
def extract_text(content: bytes, filename: str) -> str:
    fname = filename.lower()

    if fname.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages).strip()

    elif fname.endswith(".docx") or fname.endswith(".doc"):
        import tempfile, shutil
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(content); tmp.close()
        try:
            return docx2txt.process(tmp.name).strip()
        finally:
            os.unlink(tmp.name)

    elif fname.endswith(".txt"):
        return content.decode("utf-8", errors="ignore").strip()

    raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")


# ───────────────────────────────────────────
# SYSTEM PROMPT  (same structure as your original)
# ───────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI resume parser. Extract information from the resume text and return ONLY a valid JSON object.
Do NOT include markdown blocks like ```json or any conversational text. Just output raw JSON.

The JSON MUST strictly follow this structure:
{
  "personalInfo": {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedIn": ""
  },
  "summary": "",
  "skills": ["skill1", "skill2"],
  "experience": [
    {
      "title": "",
      "company": "",
      "duration": "",
      "description": ""
    }
  ],
  "education": [
    {
      "degree": "",
      "institution": "",
      "year": ""
    }
  ]
}
"""

# ───────────────────────────────────────────
# HEALTH CHECK
# ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"message": "NexusAI Resume Parser Backend is running ✅"}


# ───────────────────────────────────────────
# PARSE ENDPOINT  — matches your original /api/parse
# ───────────────────────────────────────────
@app.post("/api/parse")
async def parse_resume(
    resume: UploadFile = File(...),
):
    # Read file
    content = await resume.read()
    if not content:
        return {"success": False, "error": "Empty file received."}

    # Extract text
    try:
        text = extract_text(content, resume.filename or "resume.pdf")
    except HTTPException as e:
        return {"success": False, "error": e.detail}

    if not text:
        return {"success": False, "error": "Could not extract text. File may be image-based or corrupted."}
    if len(text) < 50:
        return {"success": False, "error": "Extracted text is too short. File may be invalid."}

    # Call LLM
    try:
        client = openai.OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text[:6000]},
            ],
            max_tokens=2000,
            temperature=0,
        )
    except openai.AuthenticationError:
        return {"success": False, "error": "Invalid API key."}
    except openai.RateLimitError:
        return {"success": False, "error": "Rate limit hit. Please try again shortly."}
    except Exception as e:
        return {"success": False, "error": f"LLM error: {str(e)}"}

    # Clean & parse JSON
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"): raw = raw[7:]
    if raw.startswith("```"):     raw = raw[3:]
    if raw.endswith("```"):       raw = raw[:-3]
    raw = raw.strip()

    try:
        parsed_data = json.loads(raw)
    except json.JSONDecodeError:
        return {"success": False, "error": "The AI model failed to return valid JSON."}

    return {"success": True, "data": parsed_data}


# ───────────────────────────────────────────
# FRONTEND  — served at /
# ───────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path) as f:
        return HTMLResponse(content=f.read())


# ───────────────────────────────────────────
# ENTRY POINT
# ───────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)