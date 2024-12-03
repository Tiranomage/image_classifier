from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dramatiq import actor, set_broker
import torch
from torchvision import models, transforms
from PIL import Image
import sqlite3
import io
import base64

app = FastAPI()

from dramatiq.brokers.redis import RedisBroker
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

with open('imagenet_classes.txt') as f:
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
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
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

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Classification Result</title>
        <link rel="stylesheet" type="text/css" href="static/styles.css">
    </head>
    <body>
        <h1>Uploaded Image</h1>
        <img src="data:image/jpeg;base64,{image_base64}" alt="Uploaded Image" width="400"/>
        <h1>Classification Result</h1>
        <p>Predicted Animal: {predicted_class}</p>
        <p>Confidence: {confidence:.2f}</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT animal, confidence FROM results WHERE task_id=?", (task_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        animal, confidence = result
        return {"animal": animal, "confidence": confidence}
    else:
        return {"status": "Processing..."}

@app.get("/")
async def root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

app.mount("/static", StaticFiles(directory="static"), name="static")