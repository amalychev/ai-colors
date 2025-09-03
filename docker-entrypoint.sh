#!/bin/bash
set -e

echo "🚀 Запуск AI Colors..."

# Проверяем наличие обученной модели
if [ ! -f "/app/models/color_palette_generator_final.pth" ]; then
    echo "⚠️  Обученная модель не найдена"
    
    # Ищем checkpoint'ы
    if ls /app/models/checkpoint_epoch_*.pth 1> /dev/null 2>&1; then
        LATEST_CHECKPOINT=$(ls -t /app/models/checkpoint_epoch_*.pth | head -n1)
        echo "📦 Найден checkpoint: $LATEST_CHECKPOINT"
        cp "$LATEST_CHECKPOINT" /app/models/color_palette_generator_final.pth
        echo "✅ Checkpoint скопирован как финальная модель"
    else
        echo "🆕 Будет создана новая необученная модель"
    fi
fi

# Проверяем структуру папок
mkdir -p /app/models /app/data /app/training_images

echo "📁 Структура проекта:"
ls -la /app/

# Запускаем приложение
echo "🎨 Запускаем AI Colors API на порту 8080..."
exec "$@"