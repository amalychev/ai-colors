import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class ColorPaletteGenerator(nn.Module):
    def __init__(self, input_dim: int = 27, hidden_dim: int = 256, max_colors: int = 10):
        """
        Нейросеть для генерации цветовых палитр
        
        Args:
            input_dim: размер входного вектора (до 9 цветов * 3 RGB канала = 27)
            hidden_dim: размер скрытых слоев
            max_colors: максимальное количество цветов в палитре
        """
        super(ColorPaletteGenerator, self).__init__()
        self.max_colors = max_colors
        self.input_dim = input_dim
        
        # Encoder - кодирует входные цвета в латентное пространство
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 для количества желаемых цветов
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU()
        )
        
        # Decoder - генерирует новые цвета из латентного пространства
        self.decoder = nn.Sequential(
            nn.Linear(128, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, max_colors * 3),  # max_colors цветов * 3 RGB канала
            nn.Sigmoid()  # Ограничиваем выход в диапазоне [0, 1]
        )
        
        # Attention mechanism для фокуса на важных цветах
        self.attention = nn.MultiheadAttention(embed_dim=3, num_heads=1, batch_first=True)
        
    def forward(self, input_colors: torch.Tensor, target_count: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети
        
        Args:
            input_colors: тензор входных цветов [batch_size, input_dim]
            target_count: количество желаемых выходных цветов [batch_size, 1]
        
        Returns:
            Сгенерированные цвета [batch_size, max_colors * 3]
        """
        batch_size = input_colors.size(0)
        
        # Объединяем входные цвета с целевым количеством
        x = torch.cat([input_colors, target_count], dim=1)
        
        # Кодируем
        encoded = self.encoder(x)
        
        # Декодируем
        decoded = self.decoder(encoded)
        
        return decoded
    
    def generate_palette(self, input_colors: List[Tuple[int, int, int]], 
                        target_count: int, device: str = 'cpu', 
                        randomize: bool = True, seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
        """
        Генерирует цветовую палитру на основе входных цветов
        
        Args:
            input_colors: список входных цветов в формате RGB (может быть пустым)
            target_count: желаемое количество выходных цветов
            device: устройство для вычислений ('cpu' или 'cuda')
            randomize: добавлять ли случайность для разнообразия результатов
            seed: фиксированный seed для воспроизводимости (если randomize=True)
        
        Returns:
            Список сгенерированных цветов
        """
        self.eval()
        
        with torch.no_grad():
            # Устанавливаем seed если нужно
            if randomize and seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            elif randomize:
                # Используем случайный seed
                import time
                random_seed = int(time.time() * 1000000) % 2**32
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
            
            # Если нет входных цветов, генерируем случайные базовые цвета
            if not input_colors:
                import random
                # Генерируем 1-3 случайных цвета для большего разнообразия
                num_base_colors = random.randint(1, 3)
                for _ in range(num_base_colors):
                    base_color = (
                        random.randint(0, 255),
                        random.randint(0, 255), 
                        random.randint(0, 255)
                    )
                    input_colors.append(base_color)
            
            # Подготавливаем входные данные
            input_tensor = self.prepare_input(input_colors, device)
            target_tensor = torch.tensor([[target_count / self.max_colors]], 
                                       dtype=torch.float32, device=device)
            
            # Добавляем случайный шум для разнообразия
            if randomize:
                # Более агрессивная рандомизация для входных цветов
                noise_scale = 0.3  # Увеличиваем шум для большего разнообразия
                noise = torch.randn_like(input_tensor) * noise_scale
                input_tensor = input_tensor + noise
                
                # Добавляем дополнительную вариативность через перестановку
                if input_tensor.size(1) > 1:
                    # Случайно перемешиваем каналы цветов для большего разнообразия
                    perm = torch.randperm(input_tensor.size(1))
                    input_tensor = input_tensor[:, perm]
                
                # Ограничиваем значения в диапазоне [0, 1]
                input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
                
                # Дополнительная рандомизация целевого количества
                target_noise = torch.randn_like(target_tensor) * 0.2
                target_tensor = target_tensor + target_noise
                target_tensor = torch.clamp(target_tensor, 0.1, 1.0)
                
                # Добавляем случайный контекстный вектор для еще большего разнообразия
                context_noise = torch.randn(input_tensor.size(0), input_tensor.size(1), device=input_tensor.device) * 0.1
                input_tensor = input_tensor + context_noise
                input_tensor = torch.clamp(input_tensor, 0.0, 1.0)
            
            # Генерируем палитру
            output = self.forward(input_tensor, target_tensor)
            
            # Принудительно ограничиваем вывод сети
            output = torch.clamp(output, 0.0, 1.0)
            
            # Проверяем на NaN/Inf
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                print("⚠️ NaN/Inf в выводе сети, заменяем")
                output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
                output = torch.clamp(output, 0.0, 1.0)
            
            # Преобразуем в RGB значения
            rgb_values = (output.squeeze() * 255).cpu().numpy().astype(int)
            rgb_values = np.clip(rgb_values, 0, 255)
            
            # Преобразуем в список цветов
            colors = []
            for i in range(0, len(rgb_values), 3):
                if i + 2 < len(rgb_values):
                    color = (int(rgb_values[i]), int(rgb_values[i+1]), int(rgb_values[i+2]))
                    colors.append(color)
            
            # Убираем дубликаты, сохраняя порядок
            unique_colors = []
            seen = set()
            for color in colors:
                if color not in seen:
                    unique_colors.append(color)
                    seen.add(color)
            
            # Перемешиваем порядок цветов для разнообразия
            if randomize and len(unique_colors) > 1:
                np.random.shuffle(unique_colors)
            
            # Если уникальных цветов недостаточно, генерируем дополнительные
            if len(unique_colors) < target_count:
                attempts = 0
                max_attempts = target_count * 10
                
                while len(unique_colors) < target_count and attempts < max_attempts:
                    # Генерируем случайный цвет с большим разнообразием
                    if unique_colors and randomize:
                        # Выбираем случайный подход к генерации
                        approach = np.random.choice(['variation', 'complementary', 'random'])
                        
                        if approach == 'variation':
                            # Небольшие вариации существующего цвета
                            base_color = unique_colors[np.random.randint(0, len(unique_colors))]
                            new_color = (
                                max(0, min(255, base_color[0] + np.random.randint(-50, 51))),
                                max(0, min(255, base_color[1] + np.random.randint(-50, 51))),
                                max(0, min(255, base_color[2] + np.random.randint(-50, 51)))
                            )
                        elif approach == 'complementary':
                            # Генерируем дополнительный цвет
                            base_color = unique_colors[np.random.randint(0, len(unique_colors))]
                            new_color = (
                                255 - base_color[0] + np.random.randint(-30, 31),
                                255 - base_color[1] + np.random.randint(-30, 31), 
                                255 - base_color[2] + np.random.randint(-30, 31)
                            )
                            new_color = (
                                max(0, min(255, new_color[0])),
                                max(0, min(255, new_color[1])),
                                max(0, min(255, new_color[2]))
                            )
                        else:
                            # Полностью случайный цвет
                            new_color = (
                                np.random.randint(0, 256),
                                np.random.randint(0, 256),
                                np.random.randint(0, 256)
                            )
                    else:
                        # Если нет цветов, генерируем случайный
                        new_color = (
                            np.random.randint(0, 256),
                            np.random.randint(0, 256),
                            np.random.randint(0, 256)
                        )
                    
                    if new_color not in seen:
                        unique_colors.append(new_color)
                        seen.add(new_color)
                    
                    attempts += 1
            
            # Сохраняем исходные входные цвета для фильтрации
            input_colors_set = set(input_colors) if input_colors else set()
            
            # Фильтруем входные цвета из результата
            filtered_colors = []
            for color in unique_colors:
                if color not in input_colors_set:
                    filtered_colors.append(color)
            
            # Если после фильтрации недостаточно цветов, генерируем дополнительные
            while len(filtered_colors) < target_count:
                attempts = 0
                max_attempts = 50
                
                while len(filtered_colors) < target_count and attempts < max_attempts:
                    # Генерируем новый цвет, избегая входные
                    new_color = (
                        np.random.randint(0, 256),
                        np.random.randint(0, 256),
                        np.random.randint(0, 256)
                    )
                    
                    # Проверяем, что цвет не является входным и уникален
                    if (new_color not in input_colors_set and 
                        new_color not in filtered_colors):
                        filtered_colors.append(new_color)
                    
                    attempts += 1
                
                # Если не удалось сгенерировать достаточно, прерываем
                if attempts >= max_attempts:
                    break
            
            # Возвращаем только нужное количество цветов
            return filtered_colors[:target_count]
    
    def prepare_input(self, colors: List[Tuple[int, int, int]], device: str = 'cpu') -> torch.Tensor:
        """
        Подготавливает входные цвета для нейросети
        """
        # Нормализуем цвета в диапазон [0, 1]
        normalized_colors = []
        for r, g, b in colors:
            # Убеждаемся что значения в правильном диапазоне
            r = max(0, min(255, int(r)))
            g = max(0, min(255, int(g)))
            b = max(0, min(255, int(b)))
            normalized_colors.extend([r/255.0, g/255.0, b/255.0])
        
        # Дополняем нулями до нужного размера
        while len(normalized_colors) < self.input_dim:
            normalized_colors.append(0.0)
        
        # Обрезаем, если слишком длинный
        normalized_colors = normalized_colors[:self.input_dim]
        
        # Создаем тензор и ограничиваем значения
        tensor = torch.tensor([normalized_colors], dtype=torch.float32, device=device)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor


class ColorHarmonyDiscriminator(nn.Module):
    """
    Дискриминатор для оценки гармоничности цветовых палитр
    """
    def __init__(self, max_colors: int = 10):
        super(ColorHarmonyDiscriminator, self).__init__()
        self.max_colors = max_colors
        
        self.network = nn.Sequential(
            nn.Linear(max_colors * 3, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Оценивает гармоничность палитры
        
        Args:
            colors: цветовая палитра [batch_size, max_colors * 3]
        
        Returns:
            Оценка гармоничности [batch_size, 1] (0-1, где 1 = гармоничная)
        """
        # Принудительно ограничиваем входные данные
        colors = torch.clamp(colors, 0.0, 1.0)
        
        # Проверяем на NaN/Inf
        if torch.any(torch.isnan(colors)) or torch.any(torch.isinf(colors)):
            colors = torch.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
            colors = torch.clamp(colors, 0.0, 1.0)
        
        return self.network(colors)


class ColorLoss(nn.Module):
    """
    Кастомная функция потерь для обучения генератора цветов
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super(ColorLoss, self).__init__()
        self.alpha = alpha  # Вес для цветовой схожести
        self.beta = beta    # Вес для разнообразия
        self.mse_loss = nn.MSELoss()
    
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потери для генерации цветов
        
        Args:
            generated: сгенерированные цвета
            target: целевые цвета
        
        Returns:
            Значение потерь
        """
        # Основные потери MSE
        similarity_loss = self.mse_loss(generated, target)
        
        # Потери для разнообразия (штрафуем за слишком похожие цвета)
        diversity_loss = self.compute_diversity_loss(generated)
        
        return self.alpha * similarity_loss + self.beta * diversity_loss
    
    def compute_diversity_loss(self, colors: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет потери разнообразия для избежания генерации похожих цветов
        """
        batch_size = colors.size(0)
        colors_reshaped = colors.view(batch_size, -1, 3)  # [batch_size, num_colors, 3]
        
        diversity_loss = 0
        num_colors = colors_reshaped.size(1)
        
        for i in range(num_colors):
            for j in range(i + 1, num_colors):
                color1 = colors_reshaped[:, i, :]
                color2 = colors_reshaped[:, j, :]
                
                # Евклидово расстояние между цветами
                distance = torch.sqrt(torch.sum((color1 - color2) ** 2, dim=1))
                
                # Штрафуем за слишком маленькое расстояние
                diversity_loss += torch.exp(-distance * 10)
        
        return diversity_loss.mean() / (num_colors * (num_colors - 1) / 2)


def calculate_color_harmony_score(colors: List[Tuple[int, int, int]]) -> float:
    """
    Вычисляет оценку гармоничности цветовой палитры на основе теории цвета
    """
    if len(colors) < 2:
        return 0.5
    
    def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Value
        v = max_val
        
        return h, s, v
    
    hsv_colors = [rgb_to_hsv(r, g, b) for r, g, b in colors]
    
    harmony_score = 0
    total_pairs = 0
    
    for i in range(len(hsv_colors)):
        for j in range(i + 1, len(hsv_colors)):
            h1, s1, v1 = hsv_colors[i]
            h2, s2, v2 = hsv_colors[j]
            
            # Разность оттенков
            hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
            
            # Проверяем классические цветовые гармонии
            # Комплементарные (180°)
            if 160 <= hue_diff <= 200:
                harmony_score += 1.0
            # Триадные (120°)
            elif 100 <= hue_diff <= 140:
                harmony_score += 0.8
            # Аналоговые (30°)
            elif hue_diff <= 50:
                harmony_score += 0.6
            # Тетрадные (90°)
            elif 70 <= hue_diff <= 110:
                harmony_score += 0.7
            else:
                harmony_score += 0.3
            
            total_pairs += 1
    
    return harmony_score / total_pairs if total_pairs > 0 else 0.5