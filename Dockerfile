# Stage 1: Build React frontend
FROM node:18-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=frontend-build /frontend/dist ./static

RUN mkdir -p uploads

EXPOSE 7860

CMD ["python", "app.py"]
