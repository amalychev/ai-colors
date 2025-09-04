import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.neural_network import ColorPaletteGenerator, ColorHarmonyDiscriminator, ColorLoss, calculate_color_harmony_score
from src.color_extractor import ColorExtractor


class ColorPaletteDataset(Dataset):
    """
    Датасет для обучения генератора цветовых палитр
    """
    def __init__(self, palettes_data: List[Dict], max_input_colors: int = 9, max_output_colors: int = 10):
        self.palettes_data = palettes_data
        self.max_input_colors = max_input_colors
        self.max_output_colors = max_output_colors
        self.samples = self.prepare_training_samples()
    
    def prepare_training_samples(self) -> List[Tuple]:
        """
        Подготавливает обучающие примеры из палитр
        """
        samples = []
        
        for palette_info in self.palettes_data:
            colors = palette_info['palette']
            
            if len(colors) < 2:
                continue
            
            # Создаем различные комбинации входных и целевых цветов
            for target_count in range(2, min(len(colors) + 1, self.max_output_colors + 1)):
                for input_count in range(1, min(len(colors), self.max_input_colors + 1)):
                    if input_count < target_count:
                        # Используем первые input_count цветов как вход
                        input_colors = colors[:input_count]
                        # Все цвета как целевые
                        target_colors = colors[:target_count]
                        
                        samples.append({
                            'input_colors': input_colors,
                            'target_colors': target_colors,
                            'target_count': target_count
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Подготавливаем входные цвета
        input_colors = self.normalize_colors(sample['input_colors'], self.max_input_colors * 3)
        target_colors = self.normalize_colors(sample['target_colors'], self.max_output_colors * 3)
        target_count = torch.tensor([sample['target_count'] / self.max_output_colors], dtype=torch.float32)
        
        return {
            'input_colors': input_colors,
            'target_colors': target_colors,
            'target_count': target_count
        }
    
    def normalize_colors(self, colors: List[Tuple[int, int, int]], target_size: int) -> torch.Tensor:
        """
        Нормализует и дополняет цвета до нужного размера
        """
        normalized = []
        for r, g, b in colors:
            # Убеждаемся что значения в правильном диапазоне
            r = max(0, min(255, r))
            g = max(0, min(255, g))  
            b = max(0, min(255, b))
            normalized.extend([r/255.0, g/255.0, b/255.0])
        
        # Дополняем нулями или обрезаем
        while len(normalized) < target_size:
            normalized.append(0.0)
        normalized = normalized[:target_size]
        
        # Убеждаемся что все значения в диапазоне [0, 1]
        tensor = torch.tensor(normalized, dtype=torch.float32)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor


class ColorPaletteTrainer:
    """
    Класс для обучения нейросети генерации цветовых палитр
    """
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Инициализация моделей
        self.generator = ColorPaletteGenerator().to(self.device)
        self.discriminator = ColorHarmonyDiscriminator().to(self.device)
        
        # Функции потерь
        self.color_loss = ColorLoss()
        self.adversarial_loss = nn.BCELoss()
        
        # Оптимизаторы
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Для отслеживания прогресса
        self.train_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'harmony_scores': []
        }
    
    def prepare_training_data(self, images_folder: str) -> List[Dict]:
        """
        Подготавливает данные для обучения из папки с изображениями
        """
        extractor = ColorExtractor()
        palettes_data = []
        
        print("Извлекаем цветовые палитры из изображений...")
        
        for filename in tqdm(os.listdir(images_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(images_folder, filename)
                try:
                    # Извлекаем палитру из изображения
                    palette = extractor.extract_palette_from_image(image_path, n_colors=8, method='kmeans')
                    
                    # Вычисляем оценку гармоничности
                    harmony_score = calculate_color_harmony_score(palette)
                    
                    # Сохраняем только гармоничные палитры (score > 0.6)
                    if harmony_score > 0.6:
                        palettes_data.append({
                            'filename': filename,
                            'palette': palette,
                            'harmony_score': harmony_score
                        })
                
                except Exception as e:
                    print(f"Ошибка обработки {filename}: {e}")
        
        print(f"Подготовлено {len(palettes_data)} цветовых палитр")
        return palettes_data
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Обучение на одной эпохе
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_harmony = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch}")):
            input_colors = batch['input_colors'].to(self.device)
            target_colors = batch['target_colors'].to(self.device)
            target_count = batch['target_count'].to(self.device)
            batch_size = input_colors.size(0)
            
            # Принудительно нормализуем ВСЕ данные
            def safe_normalize(tensor, name="tensor"):
                if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                    print(f"⚠️ NaN/Inf в {name}, заменяем на 0")
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.5, neginf=0.0)
                
                if torch.any(tensor < 0) or torch.any(tensor > 1):
                    print(f"⚠️ Некорректные значения в {name}: min={tensor.min():.3f}, max={tensor.max():.3f}")
                    tensor = torch.clamp(tensor, 0.0, 1.0)
                
                return tensor
            
            input_colors = safe_normalize(input_colors, "input_colors")
            target_colors = safe_normalize(target_colors, "target_colors") 
            target_count = safe_normalize(target_count, "target_count")
            
            # === Обучение дискриминатора ===
            self.optimizer_d.zero_grad()
            
            # Настоящие палитры
            real_labels = torch.ones(batch_size, 1, device=self.device)
            real_output = self.discriminator(target_colors)
            real_loss = self.adversarial_loss(real_output, real_labels)
            
            # Сгенерированные палитры
            fake_colors = self.generator(input_colors, target_count)
            # Принудительная нормализация сгенерированных цветов
            fake_colors = safe_normalize(fake_colors, "fake_colors")
            
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            fake_output = self.discriminator(fake_colors.detach())
            fake_loss = self.adversarial_loss(fake_output, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            self.optimizer_d.step()
            
            # === Обучение генератора ===
            self.optimizer_g.zero_grad()
            
            # Генерируем палитры
            generated_colors = self.generator(input_colors, target_count)
            # Нормализуем сгенерированные цвета
            generated_colors = safe_normalize(generated_colors, "generated_colors")
            
            # Adversarial loss
            gen_output = self.discriminator(generated_colors)
            adversarial_loss = self.adversarial_loss(gen_output, real_labels)
            
            # Color similarity loss (только MSE, без diversity loss)
            color_similarity_loss = nn.MSELoss()(generated_colors, target_colors)
            
            # Общие потери генератора
            g_loss = adversarial_loss + color_similarity_loss
            g_loss.backward()
            self.optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            
            # Вычисляем гармоничность сгенерированных палитр
            with torch.no_grad():
                gen_colors_np = (generated_colors[0] * 255).cpu().numpy().astype(int)
                colors_rgb = [(int(gen_colors_np[i]), int(gen_colors_np[i+1]), int(gen_colors_np[i+2])) 
                             for i in range(0, len(gen_colors_np)-2, 3)]
                harmony = calculate_color_harmony_score(colors_rgb[:5])  # Первые 5 цветов
                epoch_harmony += harmony
        
        return {
            'generator_loss': epoch_g_loss / len(dataloader),
            'discriminator_loss': epoch_d_loss / len(dataloader),
            'harmony_score': epoch_harmony / len(dataloader)
        }
    
    def train(self, images_folder: str, num_epochs: int = 100, batch_size: int = 32, 
              save_interval: int = 10) -> None:
        """
        Основная функция обучения
        """
        # Подготавливаем данные
        palettes_data = self.prepare_training_data(images_folder)
        
        if not palettes_data:
            raise ValueError("Нет подходящих изображений для обучения!")
        
        # Создаем датасет и даталоадер
        dataset = ColorPaletteDataset(palettes_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=False)
        
        print(f"Начинаем обучение на {len(dataset)} примерах...")
        
        for epoch in range(num_epochs):
            # Обучение на эпохе
            epoch_metrics = self.train_epoch(dataloader, epoch)
            
            # Сохраняем историю
            self.train_history['generator_loss'].append(epoch_metrics['generator_loss'])
            self.train_history['discriminator_loss'].append(epoch_metrics['discriminator_loss'])
            self.train_history['harmony_scores'].append(epoch_metrics['harmony_score'])
            
            # Очистка памяти каждые 20 эпох
            if epoch % 20 == 0 and epoch > 0:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Выводим прогресс
            if epoch % 10 == 0:
                print(f"Эпоха {epoch}:")
                print(f"  Generator Loss: {epoch_metrics['generator_loss']:.4f}")
                print(f"  Discriminator Loss: {epoch_metrics['discriminator_loss']:.4f}")
                print(f"  Harmony Score: {epoch_metrics['harmony_score']:.4f}")
            
            # Сохраняем модель
            if epoch % save_interval == 0 and epoch > 0:
                self.save_model(f"models/checkpoint_epoch_{epoch}.pth")
        
        # Сохраняем финальную модель
        self.save_model("models/color_palette_generator_final.pth")
        self.plot_training_history()
    
    def save_model(self, path: str) -> None:
        """
        Сохраняет модель и состояние обучения
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'train_history': self.train_history
        }
        
        torch.save(checkpoint, path)
        print(f"Модель сохранена: {path}")
    
    def load_model(self, path: str) -> None:
        """
        Загружает модель
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.train_history = checkpoint.get('train_history', self.train_history)
        
        print(f"Модель загружена: {path}")
    
    def plot_training_history(self) -> None:
        """
        Строит графики истории обучения
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Generator loss
        axes[0].plot(self.train_history['generator_loss'])
        axes[0].set_title('Generator Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        # Discriminator loss
        axes[1].plot(self.train_history['discriminator_loss'])
        axes[1].set_title('Discriminator Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        
        # Harmony scores
        axes[2].plot(self.train_history['harmony_scores'])
        axes[2].set_title('Harmony Scores')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


def test_generate_palette(model_path: str = "models/color_palette_generator_final.pth"):
    """
    Тестовая функция для генерации палитры
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Загружаем модель
    trainer = ColorPaletteTrainer(device)
    trainer.load_model(model_path)
    
    # Тестируем генерацию
    input_colors = [(255, 0, 0), (0, 255, 0)]  # Красный и зеленый
    target_count = 5
    
    generated_palette = trainer.generator.generate_palette(input_colors, target_count, device)
    
    print("Входные цвета:", input_colors)
    print("Сгенерированная палитра:", generated_palette)
    
    # Вычисляем оценку гармоничности
    harmony_score = calculate_color_harmony_score(generated_palette)
    print(f"Оценка гармоничности: {harmony_score:.3f}")
    
    return generated_palette