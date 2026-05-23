import gradio as gr
import openai
import cv2
import base64
from PIL import Image
import io
import tempfile

client = openai.OpenAI(
    api_key="API_KEY",
    base_url="BASE_URL"
)

def summarize_video(video):

    FRAME_INTERVAL = 1

    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_INTERVAL)

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

    content = [
        {
            "type": "text",
            "text": "These are frames extracted from a video. Summarize the full video in detail."
        }
    ]

    for frame in frames_base64[:20]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content

demo = gr.Interface(
    fn=summarize_video,
    inputs=gr.Video(),
    outputs="text",
    title="AI Video Summarizer"
)

demo.launch()