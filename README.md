# Классификатор изображений

## Описание

Веб-приложение, которое позволяет пользователям загружать изображения, сохранять их  и классифицировать изображения для идентификации животного с помощью предварительно подготовленной модели машинного обучения. Классификация выполняется асинхронно с использованием dramatiq и Redis в качестве посредника сообщений.

## Стек технологий

- Python 3.8+
- FastAPI для создания веб-сервиса.
- Dramatiq для асинхронной обработки задач.
- Redis в качестве брокера сообщений для Dramatiq.
- SQLite для хранения данных задач и результатов.
- Pillow и Torchvision для обработки и классификации изображений.
- pytest для написания и запуска тестов.

## Загрузка

1. Клонировать репозиторий:

   ```bash
   git clone https://github.com/Tiranomage/image_classifier.git
   cd image_classifier

2. Создание и активация virtualvenv:

   ```bash
   python -m venv venv
   source venv/bin/activate

3. Установка зависимостей:

   ```bash
   pip install -r requirements.txt
   
4. Запуск Redis сервера:

   ```bash
   redis-server

5. Запуск FastAPi:

   ```bash
   python -m uvicorn main:app --reload

6. Перейти на страницу с приложением:

   Открыть браузер и перейти по адресу http://127.0.0.1:8000/ для доступа к веб-интерфейсу.

## Запуск тестов

1. Запустить тесты в консоли

   ```bash
   python -m pytest tests/test_main.py
