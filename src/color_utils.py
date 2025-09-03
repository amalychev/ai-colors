"""
Утилиты для работы с цветами в различных форматах
"""

import re
from typing import List, Tuple, Union

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Конвертирует HEX цвет в RGB
    
    Args:
        hex_color: цвет в формате '#FF0000' или 'FF0000' или '#F00'
    
    Returns:
        Кортеж (R, G, B)
    """
    # Убираем # если есть
    hex_color = hex_color.lstrip('#')
    
    # Поддерживаем короткий формат (#F00 -> #FF0000)
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    
    # Проверяем корректность
    if not re.match(r'^[0-9A-Fa-f]{6}$', hex_color):
        raise ValueError(f"Некорректный HEX цвет: {hex_color}")
    
    # Конвертируем
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (r, g, b)

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Конвертирует RGB в HEX
    
    Args:
        rgb: кортеж (R, G, B)
    
    Returns:
        Цвет в формате '#FF0000'
    """
    r, g, b = rgb
    
    # Проверяем диапазон
    for component in [r, g, b]:
        if not 0 <= component <= 255:
            raise ValueError(f"RGB компонент вне диапазона 0-255: {component}")
    
    return f"#{r:02x}{g:02x}{b:02x}".upper()

def parse_colors(colors: List[str]) -> List[Tuple[int, int, int]]:
    """
    Парсит список цветов в различных форматах
    
    Args:
        colors: список цветов в форматах:
                - HEX: '#FF0000', 'FF0000', '#F00'
                - RGB: '255,0,0'
    
    Returns:
        Список кортежей (R, G, B)
    """
    parsed_colors = []
    
    for color_str in colors:
        color_str = color_str.strip()
        
        # HEX формат
        if color_str.startswith('#') or re.match(r'^[0-9A-Fa-f]{3,6}$', color_str):
            try:
                rgb = hex_to_rgb(color_str)
                parsed_colors.append(rgb)
                continue
            except ValueError:
                pass
        
        # RGB формат "255,0,0"
        if ',' in color_str:
            try:
                rgb = tuple(map(int, color_str.split(',')))
                if len(rgb) != 3:
                    raise ValueError()
                
                # Проверяем диапазон
                for component in rgb:
                    if not 0 <= component <= 255:
                        raise ValueError()
                
                parsed_colors.append(rgb)
                continue
            except ValueError:
                pass
        
        # Если ничего не подошло
        raise ValueError(f"Некорректный формат цвета: {color_str}")
    
    return parsed_colors

def format_palette(palette: List[Tuple[int, int, int]], format_type: str = 'hex') -> List[str]:
    """
    Форматирует палитру в указанный формат
    
    Args:
        palette: список кортежей (R, G, B)
        format_type: 'hex' или 'rgb'
    
    Returns:
        Список цветов в указанном формате
    """
    if format_type.lower() == 'hex':
        return [rgb_to_hex(color) for color in palette]
    elif format_type.lower() == 'rgb':
        return [f"{r},{g},{b}" for r, g, b in palette]
    else:
        raise ValueError(f"Неподдерживаемый формат: {format_type}")

def display_palette(palette: List[Tuple[int, int, int]], title: str = "Палитра"):
    """
    Красиво выводит палитру в терминал с цветной визуализацией
    """
    print(f"\n🎨 {title}:")
    
    # Показываем большие цветные блоки
    print("\n   " + "".join([create_color_block(color, wide=True) for color in palette]))
    
    # Показываем детали каждого цвета
    for i, (r, g, b) in enumerate(palette, 1):
        hex_color = rgb_to_hex((r, g, b))
        rgb_str = f"RGB({r:3}, {g:3}, {b:3})"
        
        # Цветной квадратик
        color_block = create_color_block((r, g, b))
        
        # Определяем яркость для выбора цвета текста
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        text_color = "white" if brightness < 128 else "black"
        
        print(f"  {i}. {color_block} {hex_color} | {rgb_str}")

def create_color_block(color: Tuple[int, int, int], wide: bool = False) -> str:
    """
    Создает цветной блок для терминала
    """
    r, g, b = color
    
    try:
        # ANSI escape codes для True Color (24-bit)
        bg_color = f"\033[48;2;{r};{g};{b}m"
        reset = "\033[0m"
        
        if wide:
            # Широкий блок для общего вида палитры
            return f"{bg_color}      {reset}"
        else:
            # Обычный блок для списка
            return f"{bg_color}    {reset}"
    except:
        # Fallback для терминалов без поддержки True Color
        return "■"

def display_palette_grid(palette: List[Tuple[int, int, int]], title: str = "Палитра", cols: int = 5):
    """
    Выводит палитру в виде сетки
    """
    print(f"\n🎨 {title}:")
    print()
    
    # Разбиваем на строки
    for i in range(0, len(palette), cols):
        row = palette[i:i+cols]
        
        # Рисуем большие блоки
        for _ in range(3):  # 3 строки высотой
            line = "   "
            for color in row:
                line += create_color_block(color, wide=True)
            print(line)
        
        # Подписи под блоками
        line = "   "
        for j, (r, g, b) in enumerate(row):
            hex_color = rgb_to_hex((r, g, b))
            line += f"{hex_color[:7]:<6}"
        print(line)
        print()

def display_color_comparison(colors1: List[Tuple[int, int, int]], 
                           colors2: List[Tuple[int, int, int]],
                           title1: str = "До", title2: str = "После"):
    """
    Сравнивает две палитры рядом
    """
    print(f"\n🔄 {title1} → {title2}:")
    print()
    
    # Верхняя палитра
    print(f"   {title1}:")
    print("   " + "".join([create_color_block(color, wide=True) for color in colors1]))
    
    hex_colors1 = [rgb_to_hex(color) for color in colors1]
    print("   " + "  ".join([f"{hex_color[:7]}" for hex_color in hex_colors1]))
    
    print()
    print("   ↓")
    print()
    
    # Нижняя палитра
    print(f"   {title2}:")
    print("   " + "".join([create_color_block(color, wide=True) for color in colors2]))
    
    hex_colors2 = [rgb_to_hex(color) for color in colors2]
    print("   " + "  ".join([f"{hex_color[:7]}" for hex_color in hex_colors2]))

def create_gradient_bar(color1: Tuple[int, int, int], color2: Tuple[int, int, int], width: int = 20) -> str:
    """
    Создает градиент между двумя цветами
    """
    gradient = ""
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    for i in range(width):
        # Интерполяция цветов
        t = i / (width - 1)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        
        try:
            bg_color = f"\033[48;2;{r};{g};{b}m"
            reset = "\033[0m"
            gradient += f"{bg_color} {reset}"
        except:
            gradient += "■"
    
    return gradient

def display_harmony_analysis(palette: List[Tuple[int, int, int]], harmony_score: float):
    """
    Показывает анализ гармоничности палитры
    """
    print(f"\n🎵 Анализ гармоничности:")
    
    # Визуальный индикатор оценки
    score_bar_length = 20
    filled_length = int(score_bar_length * harmony_score)
    
    # Цветная полоса оценки (от красного к зеленому)
    score_bar = ""
    for i in range(score_bar_length):
        if i < filled_length:
            # От красного к зеленому
            r = int(255 * (1 - harmony_score))
            g = int(255 * harmony_score)
            b = 0
            try:
                bg_color = f"\033[48;2;{r};{g};{b}m"
                reset = "\033[0m"
                score_bar += f"{bg_color} {reset}"
            except:
                score_bar += "■"
        else:
            score_bar += "░"
    
    print(f"   Оценка: {score_bar} {harmony_score:.3f}")
    
    # Текстовая оценка
    if harmony_score >= 0.8:
        print("   Уровень: ⭐ Отличная гармония!")
    elif harmony_score >= 0.6:
        print("   Уровень: ✅ Хорошая гармония!")
    elif harmony_score >= 0.4:
        print("   Уровень: ⚠️ Удовлетворительно")
    else:
        print("   Уровень: ❌ Слабая гармония")

def test_terminal_colors():
    """
    Тест поддержки цветов в терминале
    """
    print("🧪 Тест поддержки цветов в терминале:")
    
    # Тестовые цвета
    test_colors = [
        (255, 0, 0),    # Красный
        (0, 255, 0),    # Зеленый  
        (0, 0, 255),    # Синий
        (255, 255, 0),  # Желтый
        (255, 0, 255),  # Магента
        (0, 255, 255),  # Циан
    ]
    
    print("\n   Базовые цвета:")
    for color in test_colors:
        block = create_color_block(color)
        hex_color = rgb_to_hex(color)
        print(f"   {block} {hex_color}")
    
    print(f"\n   Градиент красный → синий:")
    print(f"   {create_gradient_bar((255, 0, 0), (0, 0, 255), 30)}")
    
    # Проверяем поддержку True Color
    import os
    colorterm = os.getenv('COLORTERM', '')
    term = os.getenv('TERM', '')
    
    print(f"\n   TERM: {term}")
    print(f"   COLORTERM: {colorterm}")
    
    if 'truecolor' in colorterm.lower() or '24bit' in colorterm.lower():
        print("   ✅ True Color поддерживается!")
    elif 'color' in term:
        print("   ⚠️ Базовые цвета поддерживаются")
    else:
        print("   ❌ Ограниченная поддержка цветов")

# Именованные цвета для удобства
NAMED_COLORS = {
    'red': '#FF0000',
    'green': '#00FF00', 
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'magenta': '#FF00FF',
    'cyan': '#00FFFF',
    'white': '#FFFFFF',
    'black': '#000000',
    'orange': '#FFA500',
    'purple': '#800080',
    'pink': '#FFC0CB',
    'brown': '#A52A2A',
    'gray': '#808080',
    'grey': '#808080'
}

def parse_named_color(color_name: str) -> Tuple[int, int, int]:
    """
    Парсит именованный цвет
    """
    color_name = color_name.lower().strip()
    if color_name in NAMED_COLORS:
        return hex_to_rgb(NAMED_COLORS[color_name])
    else:
        available = ', '.join(NAMED_COLORS.keys())
        raise ValueError(f"Неизвестный цвет '{color_name}'. Доступные: {available}")