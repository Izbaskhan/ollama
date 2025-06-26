# Используем базовый образ Python
FROM python:3.9-slim-buster

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все остальные файлы приложения
COPY . .

# Команда, которая будет выполняться при запуске контейнера
# Мы не будем запускать скрипт автоматически, а будем вызывать его вручную через docker compose run
# CMD ["python", "rag_pipeline.py"]