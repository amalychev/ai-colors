# 🎨 AI Colors - Нейросеть для генерации цветовых палитр

Интеллектуальная система для создания гармоничных цветовых палитр на основе нейронных сетей. Проект позволяет обучить модель на ваших изображениях и генерировать красивые цветовые сочетания.

## ✨ Возможности

- 🧠 **Обучение на пользовательских данных**: Обучите модель на своих изображениях дизайна
- 🎯 **Гибкая генерация**: Задайте от 1 до 9 входных цветов и получите от 2 до 10 цветов в палитре
- 🔍 **Извлечение цветов**: Автоматическое извлечение цветовых палитр из изображений
- 🌈 **Оценка гармоничности**: Автоматическая оценка красоты цветовых сочетаний
- 🚀 **REST API**: Простое HTTP API для интеграции в другие проекты
- 🎨 **Множество методов**: K-means, ColorThief, гистограммный анализ

## 🛠 Установка

### Требования
- Python 3.8+
- CUDA (опционально, для GPU ускорения)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## 📁 Структура проекта

```
ai-colors/
├── src/                    # Исходный код
│   ├── color_extractor.py  # Модуль извлечения цветов
│   ├── neural_network.py   # Архитектура нейросети
│   ├── trainer.py          # Система обучения
│   └── api.py             # REST API
├── training_images/        # Папка для обучающих изображений
├── data/                   # Обработанные данные
├── models/                 # Сохраненные модели
├── notebooks/              # Jupyter ноутбуки
├── tests/                  # Тесты
├── main.py                 # Главный скрипт
└── requirements.txt        # Зависимости
```

## 🚀 Быстрый старт

### 1. Подготовка данных

Поместите ваши изображения дизайнов в папку `training_images/`:

```bash
# Создание папки для изображений
mkdir -p training_images

# Скопируйте ваши изображения в эту папку
cp /path/to/your/images/* training_images/
```

### 2. Обработка изображений

Извлеките цветовые палитры из изображений:

```bash
python main.py process training_images
```

### 3. Обучение модели

```bash
python main.py train training_images --epochs 100 --batch-size 32
```

### 4. Запуск API

```bash
python main.py api
```

API будет доступен по адресу: http://localhost:5000

## 📖 Использование

### Командная строка

#### Извлечение цветов из изображения

```bash
# Извлечь 5 цветов методом K-means
python main.py extract path/to/image.jpg --colors 5 --method kmeans

# Использовать ColorThief
python main.py extract path/to/image.jpg --colors 8 --method colorthief
```

#### Тестирование генерации

```bash
# Генерация палитры из красного и зеленого цветов
python main.py test 255,0,0 0,255,0 --count 7
```

### HTTP API

#### Генерация палитры

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input_colors": [[255, 0, 0], [0, 255, 0]],
    "target_count": 5
  }'
```

#### Оценка гармоничности

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "colors": [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
  }'
```

#### Случайная палитра

```bash
curl http://localhost:5000/random
```

### Python API

```python
from src.neural_network import ColorPaletteGenerator
from src.trainer import ColorPaletteTrainer

# Загрузка обученной модели
trainer = ColorPaletteTrainer()
trainer.load_model("models/color_palette_generator_final.pth")

# Генерация палитры
input_colors = [(255, 0, 0), (0, 255, 0)]  # Красный и зеленый
palette = trainer.generator.generate_palette(input_colors, target_count=5)

print("Сгенерированная палитра:", palette)
```

## 🎨 Примеры палитр

### Входные цвета: Красный + Синий
- 🔴 `(255, 0, 0)` - Красный
- 🔵 `(0, 0, 255)` - Синий

### Сгенерированная палитра:
- 🔴 `(255, 0, 0)` - Красный
- 🔵 `(0, 0, 255)` - Синий  
- 🟣 `(128, 0, 128)` - Фиолетовый
- 🌸 `(255, 192, 203)` - Розовый
- 💙 `(173, 216, 230)` - Светло-голубой

**Оценка гармоничности: 0.827** ⭐

## ⚙️ Настройка

### Параметры обучения

- `--epochs`: Количество эпох обучения (по умолчанию: 100)
- `--batch-size`: Размер батча (по умолчанию: 32)
- Модель автоматически сохраняется каждые 10 эпох

### Архитектура нейросети

- **Encoder-Decoder** архитектура с attention механизмом
- **Adversarial training** с дискриминатором для оценки гармоничности
- **Custom loss function** учитывающая цветовую близость и разнообразие

## 📊 Метрики качества

Система оценивает палитры по следующим критериям:

- **Цветовая гармония**: Классические правила (комплементарные, триадные, аналоговые)
- **Разнообразие**: Достаточное различие между цветами
- **Эстетическая привлекательность**: На основе обученной модели

### Шкала оценок:
- **0.8+** - Отлично 🌟
- **0.6-0.8** - Хорошо ✅  
- **0.4-0.6** - Удовлетворительно ⚠️
- **<0.4** - Плохо ❌

## 🔧 Расширение

### Добавление новых методов извлечения цветов

```python
# В color_extractor.py
def custom_extraction_method(self, image_path: str, n_colors: int) -> List[Tuple[int, int, int]]:
    # Ваш код извлечения цветов
    return colors
```

### Кастомная функция потерь

```python
# В neural_network.py
class CustomLoss(nn.Module):
    def forward(self, generated, target):
        # Ваша логика вычисления потерь
        return loss
```

## 🤝 Вклад в проект

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменений (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📝 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 🙏 Благодарности

- PyTorch команда за отличный фреймворк
- ColorThief за алгоритмы извлечения цветов
- Сообщество дизайнеров за вдохновение

---

**Создано с ❤️ для дизайнеров и разработчиков**