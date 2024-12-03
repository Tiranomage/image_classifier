# Image Classification Web App

## Description

A web application that allows users to upload images, store them in an S3 bucket, and classify the images to identify the animal using a pre-trained machine learning model. The classification is performed asynchronously using dramatiq with Redis as the message broker.

## Tech Stack

- FastAPI for the web framework
- Dramatiq for background tasks
- Redis as the message broker
- AWS S3 for storage
- PyTorch and torchvision for image classification

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/image-classification-app.git