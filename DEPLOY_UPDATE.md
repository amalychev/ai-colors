# 🔄 Инструкция по обновлению для исправления og:image

## Проблема
Facebook Sharing Debugger показывает ошибку: "URL, указанный для og:image (https://ai-colors.online/og.jpg), невозможно обработать как изображение. Недействительный тип контента."

## Решение

### 1. На продакшн сервере выполнить обновление:
```bash
cd /path/to/ai-colors
git pull
docker-compose build --no-cache
docker-compose up -d
```

### 2. Проверить доступность изображения:
```bash
curl -I https://ai-colors.online/og.jpg
# Должен вернуть HTTP 200 с Content-Type: image/jpeg
```

### 3. Очистить кеш Facebook:
1. Перейти на https://developers.facebook.com/tools/debug/
2. Ввести URL: https://ai-colors.online/
3. Нажать "Debug"
4. Нажать "Scrape Again" для обновления кеша

## Что было исправлено:
- ✅ Добавлен маршрут `/og.jpg` для обслуживания изображения
- ✅ Добавлены маршруты `/favicons/<filename>` для favicon файлов  
- ✅ Исправлена обработка ошибок для статических файлов (возвращает 404 вместо JSON)
- ✅ Добавлены файлы favicon в проект
- ✅ Обновлены мета-теги с размерами изображения (1200×630)

## Проверка после деплоя:
```bash
# Проверка основного изображения
curl -s -o /dev/null -w "%{http_code}" https://ai-colors.online/og.jpg

# Проверка favicon файлов
curl -s -o /dev/null -w "%{http_code}" https://ai-colors.online/favicons/favicon-96x96.png
curl -s -o /dev/null -w "%{http_code}" https://ai-colors.online/favicons/favicon.svg
```

Все проверки должны возвращать HTTP 200.