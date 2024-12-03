from app.main import app, DB_NAME
from fastapi.testclient import TestClient
from dramatiq import Worker
from dramatiq.brokers.redis import RedisBroker
import sqlite3

client = TestClient(app)

def setup_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images")
    cursor.execute("DELETE FROM results")
    conn.commit()
    conn.close()

def test_upload_image():
    setup_db()
    broker = RedisBroker()
    worker = Worker(broker, worker_timeout=100)
    worker.start()

    with open("test_image.jpg", "rb") as f:
        response = client.post("/upload/", files={"file": ("test_image.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    assert "task_id" in response.json()

    task_id = response.json()["task_id"]
    import time
    time.sleep(5)
    response_result = client.get(f"/result/{task_id}")
    assert response_result.status_code == 200

    worker.stop()

if __name__ == "__main__":
    test_upload_image()