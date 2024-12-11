from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dramatiq import actor, set_broker
import torch
from torchvision import models, transforms
from PIL import Image
import sqlite3
import io
import base64
import uuid
from dramatiq.brokers.redis import RedisBroker

app = FastAPI()

broker = RedisBroker(host="localhost", port=6379)
set_broker(broker)

DB_NAME = "images.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            task_id TEXT PRIMARY KEY,
            image_data BLOB
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            task_id TEXT PRIMARY KEY,
            animal TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with open('imagenet_classes.txt', 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

@actor
def classify_image(task_id: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT image_data FROM images WHERE task_id=?", (task_id,))
    image_data = cursor.fetchone()[0]
    conn.close()

    image = Image.open(io.BytesIO(image_data))
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_class = probabilities.topk(1)
    predicted_class = classes[top_class.item()]
    confidence = top_prob.item()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO results (task_id, animal, confidence)
        VALUES (?, ?, ?)
    """, (task_id, predicted_class, confidence))
    conn.commit()
    conn.close()

@app.post("/upload/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    task_id = str(uuid.uuid4())

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO images (task_id, image_data)
        VALUES (?, ?)
    """, (task_id, image_bytes))
    conn.commit()
    conn.close()

    classify_image(task_id)

    image = Image.open(io.BytesIO(image_bytes))
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_class = probabilities.topk(1)
    predicted_class = classes[top_class.item()]
    confidence = top_prob.item()

    image_io = io.BytesIO()
    image.save(image_io, format='JPEG')
    image_base64 = base64.b64encode(image_io.getvalue()).decode('utf-8')

    with open("result.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    base_url = request.base_url
    static_url = str(base_url) + "static/styles.css"

    html_content = html_content.format(
        image_base64=image_base64,
        predicted_class=predicted_class,
        confidence=confidence,
        static_url=static_url
    )

    return HTMLResponse(content=html_content)

@app.get("/")
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

app.mount("/static", StaticFiles(directory="static"), name="static")