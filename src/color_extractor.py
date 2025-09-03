import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from colorthief import ColorThief
import cv2
from typing import List, Tuple
import os

class ColorExtractor:
    def __init__(self):
        pass
    
    def extract_colors_kmeans(self, image_path: str, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Извлекает доминирующие цвета из изображения используя K-means кластеризацию
        """
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((150, 150))
        
        # Преобразуем изображение в массив пикселей
        pixels = np.array(image).reshape(-1, 3)
        
        # Применяем K-means кластеризацию
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Получаем центры кластеров как доминирующие цвета
        colors = kmeans.cluster_centers_.astype(int)
        
        return [tuple(color) for color in colors]
    
    def extract_colors_colorthief(self, image_path: str, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Извлекает цвета используя библиотеку ColorThief
        """
        color_thief = ColorThief(image_path)
        palette = color_thief.get_palette(color_count=n_colors)
        return palette
    
    def extract_colors_histogram(self, image_path: str, n_colors: int = 5) -> List[Tuple[int, int, int]]:
        """
        Извлекает цвета на основе гистограммы цветов
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Уменьшаем размер изображения для ускорения
        image = cv2.resize(image, (150, 150))
        
        # Преобразуем в одномерный массив
        pixels = image.reshape(-1, 3)
        
        # Применяем K-means
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    
    def rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """
        Конвертирует RGB в LAB цветовое пространство для лучшего сравнения цветов
        """
        r, g, b = [x / 255.0 for x in rgb]
        
        # Применяем gamma коррекцию
        def gamma_correction(c):
            if c > 0.04045:
                return pow((c + 0.055) / 1.055, 2.4)
            else:
                return c / 12.92
        
        r = gamma_correction(r)
        g = gamma_correction(g)
        b = gamma_correction(b)
        
        # Конвертируем в XYZ
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Нормализуем для D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # Применяем функцию Lab
        def lab_function(t):
            if t > 0.008856:
                return pow(t, 1/3)
            else:
                return (7.787 * t) + (16/116)
        
        fx = lab_function(x)
        fy = lab_function(y)
        fz = lab_function(z)
        
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return (L, a, b)
    
    def extract_palette_from_image(self, image_path: str, n_colors: int = 5, method: str = 'kmeans') -> List[Tuple[int, int, int]]:
        """
        Основной метод для извлечения цветовой палитры из изображения
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")
        
        if method == 'kmeans':
            return self.extract_colors_kmeans(image_path, n_colors)
        elif method == 'colorthief':
            return self.extract_colors_colorthief(image_path, n_colors)
        elif method == 'histogram':
            return self.extract_colors_histogram(image_path, n_colors)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def process_training_images(self, images_folder: str, output_file: str = 'data/color_palettes.txt'):
        """
        Обрабатывает все изображения в папке и сохраняет извлеченные палитры
        """
        palettes = []
        
        for filename in os.listdir(images_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(images_folder, filename)
                try:
                    palette = self.extract_palette_from_image(image_path, n_colors=8)
                    palettes.append({
                        'filename': filename,
                        'palette': palette
                    })
                except Exception as e:
                    print(f"Ошибка обработки {filename}: {e}")
        
        # Сохраняем результаты
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for palette_data in palettes:
                f.write(f"{palette_data['filename']}: {palette_data['palette']}\n")
        
        return palettes