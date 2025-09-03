# 🚀 Развертывание AI Colors в Docker

## Быстрый старт

### 1. Клонирование проекта на сервер
```bash
git clone <your-repo-url> ai-colors
cd ai-colors
```

### 2. Копирование обученной модели (если есть)
```bash
# Если у вас есть обученная модель локально
scp models/color_palette_generator_final.pth user@server:/path/to/ai-colors/models/

# Или скопируйте checkpoint
scp models/checkpoint_epoch_*.pth user@server:/path/to/ai-colors/models/
```

### 3. Запуск приложения
```bash
# Простой запуск
docker-compose up -d

# Или с Nginx (для продакшена)
docker-compose --profile production up -d
```

### 4. Проверка работы
```bash
curl http://localhost:8080/health
```

## Варианты развертывания

### Вариант 1: Только приложение (разработка)
```bash
docker-compose up ai-colors
```
Доступно на: http://server-ip:8080

### Вариант 2: С Nginx прокси (продакшен)
```bash
docker-compose --profile production up -d
```
Доступно на: http://server-ip:80

### Вариант 3: Кастомная сборка
```bash
# Сборка образа
docker build -t ai-colors:latest .

# Запуск контейнера
docker run -d \
  --name ai-colors \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/training_images:/app/training_images \
  ai-colors:latest
```

## Управление

### Просмотр логов
```bash
docker-compose logs -f ai-colors
```

### Остановка
```bash
docker-compose down
```

### Перезапуск
```bash
docker-compose restart ai-colors
```

### Обновление
```bash
git pull
docker-compose build --no-cache
docker-compose up -d
```

## Обучение модели в контейнере

### Подготовка изображений
```bash
# Скопируйте изображения в training_images/
docker cp /local/path/images ai-colors-app:/app/training_images/
```

### Обучение
```bash
# Зайдите в контейнер
docker exec -it ai-colors-app bash

# Запустите обучение
python main.py train training_images --epochs 50 --batch-size 16

# Или из хоста
docker exec ai-colors-app python main.py train training_images --epochs 50
```

## Переменные окружения

Создайте файл `.env`:
```env
# Порт приложения
APP_PORT=8080

# Устройство для вычислений
DEVICE=cpu

# Уровень логирования
LOG_LEVEL=INFO

# Максимальное время ожидания
TIMEOUT=300
```

Затем запустите:
```bash
docker-compose --env-file .env up -d
```

## Мониторинг

### Healthcheck
```bash
curl http://localhost:8080/health
```

### Статус контейнера
```bash
docker ps
docker stats ai-colors-app
```

### Использование ресурсов
```bash
docker exec ai-colors-app python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'RAM: {psutil.virtual_memory().percent}%')
"
```

## Безопасность

### Настройка firewall
```bash
# Разрешаем только нужные порты
sudo ufw allow 80
sudo ufw allow 443
sudo ufw deny 8080  # Закрываем прямой доступ к приложению
```

### SSL сертификат (с Let's Encrypt)
```bash
# Установите certbot
sudo apt install certbot python3-certbot-nginx

# Получите сертификат
sudo certbot --nginx -d your-domain.com

# Обновите nginx.conf для HTTPS
```

## Резервное копирование

### Модели
```bash
# Создание backup
docker cp ai-colors-app:/app/models ./backup/models-$(date +%Y%m%d)

# Восстановление
docker cp ./backup/models-20241201/ ai-colors-app:/app/models/
```

### Полное резервное копирование
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backup/ai-colors-$DATE"

mkdir -p $BACKUP_DIR
docker cp ai-colors-app:/app/models $BACKUP_DIR/
docker cp ai-colors-app:/app/data $BACKUP_DIR/
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR
```

## Масштабирование

### Увеличение ресурсов
```yaml
# В docker-compose.yml
services:
  ai-colors:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

### Несколько инстансов
```bash
# Запуск дополнительных экземпляров
docker-compose up --scale ai-colors=3
```

## Troubleshooting

### Проблемы с памятью
```bash
# Увеличьте память в docker-compose.yml
services:
  ai-colors:
    mem_limit: 4g
```

### Медленная генерация
```bash
# Используйте GPU если доступен
docker run --gpus all ...
```

### Логи приложения
```bash
docker-compose logs ai-colors | grep ERROR
```

### Очистка
```bash
# Удалить неиспользуемые образы
docker system prune -a

# Очистить volumes
docker volume prune
```