.PHONY: help build up down logs restart clean train test

# Переменные
DOCKER_COMPOSE = docker-compose
APP_NAME = ai-colors-app

help: ## Показать эту справку
	@echo "🎨 AI Colors - Docker команды:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Собрать Docker образ
	$(DOCKER_COMPOSE) build --no-cache

up: ## Запустить приложение
	$(DOCKER_COMPOSE) up -d
	@echo "🚀 AI Colors запущен на http://localhost:8080"

up-prod: ## Запустить с Nginx (продакшен)
	$(DOCKER_COMPOSE) --profile production up -d
	@echo "🚀 AI Colors запущен на http://localhost:80 (через Nginx)"

down: ## Остановить приложение
	$(DOCKER_COMPOSE) down

logs: ## Показать логи
	$(DOCKER_COMPOSE) logs -f ai-colors

restart: ## Перезапустить приложение
	$(DOCKER_COMPOSE) restart ai-colors

status: ## Показать статус контейнеров
	$(DOCKER_COMPOSE) ps

shell: ## Зайти в контейнер
	docker exec -it $(APP_NAME) bash

train: ## Запустить обучение в контейнере (нужно подготовить training_images/)
	docker exec $(APP_NAME) python main.py train training_images --epochs 50 --batch-size 16

test-api: ## Протестировать API
	curl -X POST http://localhost:8080/generate \
		-H "Content-Type: application/json" \
		-d '{"input_colors": [[255, 0, 0], [0, 255, 0]], "target_count": 5}'

health: ## Проверить здоровье приложения
	curl http://localhost:8080/health

clean: ## Очистить Docker ресурсы
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

backup: ## Создать резервную копию моделей
	@mkdir -p backup
	docker cp $(APP_NAME):/app/models ./backup/models-$$(date +%Y%m%d_%H%M%S)
	@echo "✅ Backup создан в backup/"

restore: ## Восстановить модели из backup (использование: make restore BACKUP=models-20241201_120000)
	@if [ -z "$(BACKUP)" ]; then echo "❌ Укажите BACKUP=folder"; exit 1; fi
	docker cp ./backup/$(BACKUP) $(APP_NAME):/app/models/
	docker restart $(APP_NAME)
	@echo "✅ Модели восстановлены из backup/$(BACKUP)"

# Полезные алиасы
start: up ## Алиас для up
stop: down ## Алиас для down