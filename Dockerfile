FROM python:3.11

WORKDIR /app

COPY backend/requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]