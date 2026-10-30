# Assignment 3: MNIST GAN Image Generation

This project trains a Generative Adversarial Network (GAN) on the MNIST dataset and adds a digit-generation endpoint to a FastAPI application.

## Generate a digit

Run the application with Docker:

```bash
docker build -t applied-gen-ai-assignment3 .
docker run --rm -p 8002:8000 applied-gen-ai-assignment3
