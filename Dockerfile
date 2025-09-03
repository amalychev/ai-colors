# Используем официальный Python образ с поддержкой машинного обучения
FROM python:3.11-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Создаем необходимые директории
RUN mkdir -p models data training_images

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Открываем порт
EXPOSE 8080

# Создаем non-root пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Копируем и настраиваем entrypoint скрипт
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/

# Устанавливаем entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Команда по умолчанию - запуск API сервера
CMD ["python", "main.py", "api", "--port", "8080"]