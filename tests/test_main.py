import pytest
from fastapi.testclient import TestClient
from app.main import app, init_db
from dramatiq import get_broker, set_broker
from dramatiq.brokers.stub import StubBroker
from unittest.mock import MagicMock
import sqlite3
import io
import torch
from PIL import Image
import os
from unittest.mock import patch

@pytest.fixture
def mock_broker():
    broker = StubBroker()
    set_broker(broker)
    yield broker
    set_broker(None)

@pytest.fixture
def temp_db():
    temp_db_file = "images.db"
    init_db()
    yield temp_db_file
    os.remove(temp_db_file)

@pytest.fixture
def client(temp_db, mock_broker):
    with patch('app.main.DB_NAME', temp_db):
        with TestClient(app) as c:
            yield c

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Загрузить изображение" in response.text

def test_upload_endpoint(client, temp_db, mock_broker):
    image = Image.new('RGB', (224, 224))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    with patch('app.main.model', MagicMock()) as mock_model:
        mock_output = torch.tensor([[0.85, 0.15]])
        mock_model.return_value = mock_output
        response = client.post(
            "/upload/",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        assert "Результат классификации" in response.text
        assert "Предсказание:" in response.text
        assert "Вероятность:" in response.text
    
    broker = get_broker()
    broker.flush_all()
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM images")
    image_count = cursor.fetchone()[0]
    assert image_count == 1
    cursor.execute("SELECT COUNT(*) FROM results")
    result_count = cursor.fetchone()[0]
    assert result_count == 1
    conn.close()