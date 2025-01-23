# Используем базовый образ с Python
FROM python:3.9-slim

# Устанавливаем зависимости для системы (если нужно)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . /app

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Указываем порт, который будет использовать Gradio
EXPOSE 7860

# Устанавливаем переменную окружения для Gradio
ENV GRADIO_SERVER_PORT=7860

# Запускаем приложение
CMD ["python", "app.py"]
