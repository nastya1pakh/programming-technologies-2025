# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt (если есть) и сам скрипт
COPY requirements.txt .
COPY example_usage.py .

# Устанавливаем зависимости (если есть)
RUN pip install --no-cache-dir -r requirements.txt

# Запускаем скрипт
CMD ["python", "example_usage.py"]