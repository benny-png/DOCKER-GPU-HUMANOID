FROM ubuntu:latest

LABEL authors="humanoid robot"

FROM python:3.10-slim

WORKDIR /app

COPY . /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80" ]

ENTRYPOINT [ "top", "-b" ]