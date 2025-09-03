#!/usr/bin/env python3
"""
Главный скрипт для запуска различных режимов AI Colors
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional

# Добавляем путь к src в PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trainer import ColorPaletteTrainer, test_generate_palette
from src.api import app, load_model
from src.color_extractor import ColorExtractor
from src.neural_network import calculate_color_harmony_score
from src.color_utils import (parse_colors, format_palette, display_palette, parse_named_color, NAMED_COLORS,
                           display_color_comparison, display_harmony_analysis, display_palette_grid, test_terminal_colors)

def train_model(images_folder: str, epochs: int = 100, batch_size: int = 32):
    """
    Запускает обучение модели
    """
    print(f"🎯 Начинаем обучение модели на изображениях из: {images_folder}")
    print(f"Эпохи: {epochs}, Размер батча: {batch_size}")
    
    if not os.path.exists(images_folder):
        print(f"❌ Папка с изображениями не найдена: {images_folder}")
        return
    
    trainer = ColorPaletteTrainer()
    try:
        trainer.train(
            images_folder=images_folder,
            num_epochs=epochs,
            batch_size=batch_size,
            save_interval=10
        )
        print("✅ Обучение завершено успешно!")
    except Exception as e:
        print(f"❌ Ошибка во время обучения: {e}")

def run_api(port: int = 8080):
    """
    Запускает API сервер
    """
    print("🚀 Запуск AI Colors API сервера...")
    print(f"🎨 Веб-интерфейс доступен по адресу: http://localhost:{port}")
    print(f"📚 API документация доступна по адресу: http://localhost:{port}/api")
    
    # Предзагружаем модель
    load_model()
    
    # Запуск Flask сервера
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )

def test_generation(input_colors: List[str], target_count: int = 5, output_format: str = 'hex', 
                   randomize: bool = True, seed: Optional[int] = None, variations: int = 1):
    """
    Тестирует генерацию палитры
    """
    print("🧪 Тестирование генерации палитры...")
    
    # Парсим входные цвета (поддерживаем HEX, RGB и именованные цвета)
    try:
        parsed_colors = []
        for color_str in input_colors:
            # Пробуем именованный цвет
            if color_str.lower() in NAMED_COLORS:
                parsed_colors.append(parse_named_color(color_str))
            else:
                # Парсим как HEX или RGB
                parsed_colors.extend(parse_colors([color_str]))
        
    except ValueError as e:
        print(f"❌ {e}")
        print(f"💡 Поддерживаемые форматы:")
        print(f"   • HEX: #FF0000, FF0000, #F00")
        print(f"   • RGB: 255,0,0")
        print(f"   • Именованные: {', '.join(list(NAMED_COLORS.keys())[:8])}...")
        return
    
    # Показываем входные цвета
    display_palette(parsed_colors, "Входные цвета")
    print(f"🎯 Целевое количество цветов: {target_count}")
    print(f"📤 Формат вывода: {output_format.upper()}")
    print(f"🎲 Рандомизация: {'включена' if randomize else 'выключена'}")
    if seed is not None:
        print(f"🌱 Seed: {seed}")
    if variations > 1:
        print(f"🔄 Вариаций: {variations}")
    
    try:
        # Используем тестовую генерацию
        from src.trainer import ColorPaletteTrainer
        
        trainer = ColorPaletteTrainer()
        trainer.load_model("models/color_palette_generator_final.pth")
        
        # Генерируем несколько вариаций
        all_palettes = []
        all_scores = []
        
        for i in range(variations):
            if variations > 1:
                print(f"\n{'='*50}")
                print(f"🎨 Вариация #{i+1}")
                print(f"{'='*50}")
            
            # Используем разные seeds для каждой вариации
            current_seed = seed + i if seed is not None else None
            
            generated_palette = trainer.generator.generate_palette(
                parsed_colors, target_count, 
                randomize=randomize, seed=current_seed
            )
            
            all_palettes.append(generated_palette)
            harmony_score = calculate_color_harmony_score(generated_palette)
            all_scores.append(harmony_score)
            
            if variations == 1:
                # Показываем сравнение для одной палитры
                display_color_comparison(parsed_colors, generated_palette, "Входные цвета", "Сгенерированная палитра")
                display_palette(generated_palette, "Детали палитры")
            else:
                # Компактный вид для множественных вариаций
                display_palette(generated_palette, f"Вариация #{i+1}")
            
            # Выводим в запрошенном формате
            formatted_colors = format_palette(generated_palette, output_format)
            print(f"\n📋 Результат #{i+1} в {output_format.upper()} формате:")
            for j, color in enumerate(formatted_colors, 1):
                print(f"   {j}. {color}")
            
            # Анализ гармоничности
            display_harmony_analysis(generated_palette, harmony_score)
        
        # Если несколько вариаций, показываем сравнение
        if variations > 1:
            print(f"\n{'='*50}")
            print("📊 Сравнение всех вариаций")
            print(f"{'='*50}")
            
            best_idx = max(range(len(all_scores)), key=lambda i: all_scores[i])
            print(f"🏆 Лучшая по гармонии: Вариация #{best_idx + 1} (оценка: {all_scores[best_idx]:.3f})")
            
            # Показываем все палитры рядом
            for i, (palette, score) in enumerate(zip(all_palettes, all_scores)):
                marker = "🏆" if i == best_idx else f"{i+1}."
                print(f"\n   {marker} Вариация #{i+1} (гармония: {score:.3f}):")
                print("   " + "".join([f"\033[48;2;{r};{g};{b}m      \033[0m" for r, g, b in palette]))
        
    except Exception as e:
        print(f"❌ Ошибка генерации: {e}")

def extract_colors(image_path: str, n_colors: int = 5, method: str = 'kmeans'):
    """
    Извлекает цвета из изображения
    """
    print(f"🎨 Извлечение {n_colors} цветов из изображения: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    extractor = ColorExtractor()
    try:
        colors = extractor.extract_palette_from_image(image_path, n_colors, method)
        print(f"✅ Извлеченные цвета: {colors}")
        
        harmony_score = calculate_color_harmony_score(colors)
        print(f"🎵 Оценка гармоничности: {harmony_score:.3f}")
        
        # Красиво выводим палитру
        display_palette(colors, f"Извлеченная палитра ({method})")
        
        # Дополнительно выводим в HEX формате
        hex_colors = format_palette(colors, 'hex')
        print(f"\n📋 HEX формат: {hex_colors}")
        
    except Exception as e:
        print(f"❌ Ошибка извлечения цветов: {e}")

def process_training_folder(images_folder: str):
    """
    Обрабатывает папку с изображениями и извлекает цветовые палитры
    """
    print(f"📁 Обработка изображений в папке: {images_folder}")
    
    if not os.path.exists(images_folder):
        print(f"❌ Папка не найдена: {images_folder}")
        return
    
    extractor = ColorExtractor()
    try:
        palettes = extractor.process_training_images(images_folder)
        print(f"✅ Обработано {len(palettes)} изображений")
        print("📄 Результаты сохранены в data/color_palettes.txt")
        
        # Показываем статистику
        total_harmony = sum(calculate_color_harmony_score(p['palette']) for p in palettes)
        avg_harmony = total_harmony / len(palettes) if palettes else 0
        print(f"📊 Средняя оценка гармоничности: {avg_harmony:.3f}")
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")

def main():
    parser = argparse.ArgumentParser(description='🎨 AI Colors - Нейросеть для генерации цветовых палитр')
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда обучения
    train_parser = subparsers.add_parser('train', help='Обучить модель на изображениях')
    train_parser.add_argument('images_folder', help='Путь к папке с изображениями для обучения')
    train_parser.add_argument('--epochs', type=int, default=100, help='Количество эпох (по умолчанию: 100)')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Размер батча (по умолчанию: 32)')
    
    # Команда запуска API
    api_parser = subparsers.add_parser('api', help='Запустить API сервер')
    api_parser.add_argument('--port', type=int, default=8080, help='Порт для сервера (по умолчанию: 8080)')
    
    # Команда тестирования
    test_parser = subparsers.add_parser('test', help='Тестировать генерацию палитры')
    test_parser.add_argument('colors', nargs='+', help='Входные цвета (HEX: #FF0000, RGB: 255,0,0, или red)')
    test_parser.add_argument('--count', type=int, default=5, help='Количество цветов для генерации')
    test_parser.add_argument('--format', choices=['hex', 'rgb'], default='hex', help='Формат вывода')
    test_parser.add_argument('--no-random', action='store_true', help='Отключить рандомизацию (детерминированный результат)')
    test_parser.add_argument('--seed', type=int, help='Фиксированный seed для воспроизводимости')
    test_parser.add_argument('--variations', type=int, default=1, help='Количество вариаций для генерации (1-10)')
    
    # Команда извлечения цветов
    extract_parser = subparsers.add_parser('extract', help='Извлечь цвета из изображения')
    extract_parser.add_argument('image_path', help='Путь к изображению')
    extract_parser.add_argument('--colors', type=int, default=5, help='Количество цветов для извлечения')
    extract_parser.add_argument('--method', choices=['kmeans', 'colorthief', 'histogram'], 
                               default='kmeans', help='Метод извлечения цветов')
    
    # Команда обработки папки
    process_parser = subparsers.add_parser('process', help='Обработать папку с изображениями')
    process_parser.add_argument('images_folder', help='Путь к папке с изображениями')
    
    # Команда тестирования терминала
    test_colors_parser = subparsers.add_parser('test-colors', help='Проверить поддержку цветов в терминале')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🎨 AI Colors - Генератор цветовых палитр")
    print("=" * 50)
    
    if args.command == 'train':
        train_model(args.images_folder, args.epochs, args.batch_size)
    
    elif args.command == 'api':
        run_api(args.port)
    
    elif args.command == 'test':
        randomize = not args.no_random
        variations = max(1, min(10, args.variations))  # Ограничиваем 1-10
        test_generation(args.colors, args.count, args.format, randomize, args.seed, variations)
    
    elif args.command == 'extract':
        extract_colors(args.image_path, args.colors, args.method)
    
    elif args.command == 'process':
        process_training_folder(args.images_folder)
    
    elif args.command == 'test-colors':
        test_terminal_colors()

if __name__ == '__main__':
    main()