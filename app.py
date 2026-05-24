import openai
import cv2
import base64
from PIL import Image
import io
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(
    api_key="TUMHARI_API_KEY",
    base_url="TUMHARA_BASE_URL"
)

@app.get("/")
def root():
    return {"status": "Video Summarizer API running!"}

@app.post("/summarize")
async def summarize(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 1)
    frames_base64 = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            frames_base64.append(img_b64)
        frame_count += 1

    cap.release()
    os.unlink(tmp_path)

    content = [{"type": "text", "text": "These are frames from a video. Summarize it in detail."}]
    for frame in frames_base64[:20]:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": content}],
        max_tokens=2000
    )

    return {"summary": response.choices[0].message.content}