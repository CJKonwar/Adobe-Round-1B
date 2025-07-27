FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY documentIntellligence.py .
COPY extraction_1A.py .
COPY main.py .
COPY requirements.txt .

COPY models ./models

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
