from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import os
from typing import List, Tuple, Dict, Any
import json
from src.neural_network import ColorPaletteGenerator, calculate_color_harmony_score
from src.trainer import ColorPaletteTrainer

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Глобальная переменная для загруженной модели
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path: str = "models/color_palette_generator_final.pth"):
    """
    Загружает обученную модель
    """
    global model
    if model is None:
        try:
            trainer = ColorPaletteTrainer(device)
            trainer.load_model(model_path)
            model = trainer.generator
            model.eval()
            print(f"Модель загружена с устройства: {device}")
        except Exception as e:
            print(f"Ошибка загрузки основной модели: {e}")
            
            # Попробуем загрузить checkpoint
            import glob
            checkpoints = glob.glob("models/checkpoint_epoch_*.pth")
            if checkpoints:
                latest_checkpoint = max(checkpoints)
                try:
                    trainer = ColorPaletteTrainer(device)
                    trainer.load_model(latest_checkpoint)
                    model = trainer.generator
                    model.eval()
                    print(f"Загружен checkpoint: {latest_checkpoint}")
                except Exception as e2:
                    print(f"Ошибка загрузки checkpoint: {e2}")
                    # Создаем новую модель
                    model = ColorPaletteGenerator().to(device)
                    model.eval()
                    print("Создана новая необученная модель")
            else:
                # Создаем новую модель если нет checkpoints
                model = ColorPaletteGenerator().to(device)
                model.eval()
                print("Создана новая необученная модель")
    return model

def validate_color(color: Any) -> bool:
    """
    Проверяет корректность цвета
    """
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        return False
    
    for component in color:
        if not isinstance(component, int) or component < 0 or component > 255:
            return False
    
    return True

def validate_input(data: Dict) -> Tuple[bool, str]:
    """
    Validates input data
    """
    # Check required fields
    if 'target_count' not in data:
        return False, "Missing field 'Number of colors in palette'"
    
    input_colors = data.get('input_colors', [])
    target_count = data['target_count']
    
    # Check input colors (now completely optional, can be empty)
    if not isinstance(input_colors, list):
        return False, "Input colors must be a list"
    
    if len(input_colors) > 9:
        return False, "Maximum 9 input colors allowed"
    
    for i, color in enumerate(input_colors):
        if not validate_color(color):
            return False, f"Invalid color at position {i}: {color}"
    
    # Check target count
    if not isinstance(target_count, int) or target_count < 2 or target_count > 10:
        return False, "Number of colors in palette must be an integer from 2 to 10"
    
    return True, ""

@app.route('/')
def index():
    """
    Главная страница с веб-интерфейсом
    """
    try:
        return render_template('index.html')
    except:
        # Fallback: встроенная HTML страница
        return """
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🎨 AI Colors - Генератор цветовых палитр на основе ИИ</title>
            <meta name="description" content="Умный генератор гармоничных цветовых палитр на основе нейросетей. Создавайте красивые цветовые схемы для дизайна за секунды.">
            <meta name="keywords" content="цветовые палитры, генератор цветов, дизайн, нейросети, ИИ, цветовая схема">
            <meta name="author" content="AI Colors">
            
            <!-- Open Graph / Facebook -->
            <meta property="og:type" content="website">
            <meta property="og:url" content="https://ai-colors.app/">
            <meta property="og:title" content="🎨 AI Colors - Генератор цветовых палитр на основе ИИ">
            <meta property="og:description" content="Умный генератор гармоничных цветовых палитр на основе нейросетей. Создавайте красивые цветовые схемы для дизайна за секунды.">
            <meta property="og:image" content="https://ai-colors.app/og-image.png">
            
            <!-- Twitter -->
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="https://ai-colors.app/">
            <meta property="twitter:title" content="🎨 AI Colors - Генератор цветовых палитр на основе ИИ">
            <meta property="twitter:description" content="Умный генератор гармоничных цветовых палитр на основе нейросетей.">
            <meta property="twitter:image" content="https://ai-colors.app/og-image.png">
            
            <script src="https://cdn.tailwindcss.com"></script>
            <script>
                tailwind.config = {
                    theme: {
                        extend: {
                            animation: {
                                'gradient': 'gradient 15s ease infinite',
                                'pulse-glow': 'pulse-glow 2s ease-in-out infinite alternate',
                                'bounce-in': 'bounce-in 0.5s ease-out'
                            },
                            keyframes: {
                                'gradient': {
                                    '0%, 100%': {
                                        'background-size': '200% 200%',
                                        'background-position': 'left center'
                                    },
                                    '50%': {
                                        'background-size': '200% 200%',
                                        'background-position': 'right center'
                                    }
                                },
                                'pulse-glow': {
                                    'from': { 'box-shadow': '0 0 20px #667eea' },
                                    'to': { 'box-shadow': '0 0 30px #764ba2, 0 0 40px #667eea' }
                                },
                                'bounce-in': {
                                    '0%': { transform: 'scale(0.3)', opacity: '0' },
                                    '50%': { transform: 'scale(1.1)' },
                                    '70%': { transform: 'scale(0.9)' },
                                    '100%': { transform: 'scale(1)', opacity: '1' }
                                }
                            }
                        }
                    }
                }
            </script>
        </head>
        <body class="min-h-screen bg-gradient-to-br from-purple-400 via-pink-500 to-red-500 animate-gradient bg-[length:200%_200%]">
            <div class="container mx-auto px-6 py-12 max-w-4xl">
                <div class="bg-white/90 backdrop-blur-lg rounded-3xl shadow-2xl p-8 md:p-12">
                    <!-- Заголовок -->
                    <div class="text-center mb-12">
                        <h1 class="text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
                            🎨 AI Colors
                        </h1>
                        <p class="text-xl text-gray-600 max-w-2xl mx-auto">
                            Генератор гармоничных цветовых палитр на основе нейросетей
                        </p>
                        <div class="mt-4 flex flex-wrap justify-center gap-2">
                            <span class="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-sm font-medium">🤖 ИИ</span>
                            <span class="px-3 py-1 bg-pink-100 text-pink-700 rounded-full text-sm font-medium">🎨 Дизайн</span>
                            <span class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">⚡ Быстро</span>
                        </div>
                    </div>
                    
                    <!-- Инструкция -->
                    <div class="mb-8 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl border border-blue-100">
                        <div class="flex items-center mb-3">
                            <div class="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center mr-3">
                                <span class="text-white text-sm">💡</span>
                            </div>
                            <h3 class="text-lg font-semibold text-gray-800">Как использовать</h3>
                        </div>
                        <p class="text-gray-600 leading-relaxed">
                            Просто выберите количество цветов и нажмите "Генерировать" для создания случайной гармоничной палитры. 
                            Или добавьте свои цвета для создания палитры на их основе.
                        </p>
                    </div>
                    
                    <!-- Цветовые входы -->
                    <div class="mb-8">
                        <div class="flex justify-between items-center mb-4">
                            <h3 class="text-lg font-semibold text-gray-700">🎯 Базовые цвета (необязательно)</h3>
                            <button onclick="clearColors()" class="text-sm text-gray-500 hover:text-red-500 transition-colors">
                                Очистить все
                            </button>
                        </div>
                        <div class="color-inputs flex flex-wrap gap-4 justify-center" id="colorInputs">
                            <button onclick="addColor()" 
                                    class="w-16 h-16 md:w-20 md:h-20 border-3 border-dashed border-gray-300 rounded-2xl 
                                           hover:border-purple-400 hover:bg-purple-50 transition-all duration-200 
                                           flex items-center justify-center text-2xl text-gray-400 hover:text-purple-500"
                                    title="Добавить цвет">
                                +
                            </button>
                        </div>
                    </div>
                    
                    <!-- Элементы управления -->
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                        <div class="space-y-2">
                            <label class="block text-sm font-semibold text-gray-700">🔢 Количество цветов</label>
                            <input type="number" id="targetCount" min="2" max="10" value="5" 
                                   class="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-all">
                        </div>
                        <div class="flex items-end">
                            <button onclick="generatePalette()" 
                                    class="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 
                                           text-white font-bold py-3 px-6 rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 
                                           transition-all duration-200 animate-pulse-glow">
                                ✨ Генерировать палитру
                            </button>
                        </div>
                    </div>
                    
                    <!-- Загрузка -->
                    <div class="loading hidden text-center py-8" id="loading">
                        <div class="inline-flex items-center">
                            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mr-3"></div>
                            <span class="text-lg text-purple-600 font-medium">Генерируем магию цветов...</span>
                        </div>
                    </div>
                    
                    <!-- Ошибки -->
                    <div class="error hidden bg-red-50 border-l-4 border-red-400 p-4 rounded-lg mb-6" id="error">
                        <div class="flex items-center">
                            <span class="text-red-400 mr-2">❌</span>
                            <span class="text-red-700 font-medium" id="errorText"></span>
                        </div>
                    </div>
                    
                    <!-- Результаты -->
                    <div class="results hidden" id="results">
                        <div class="text-center mb-6">
                            <h3 class="text-2xl font-bold text-gray-800 mb-2">🌈 Ваша палитра готова!</h3>
                            <p class="text-gray-600">Нажмите на любой цвет, чтобы скопировать его код</p>
                        </div>
                        
                        <div class="palette-display grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6" id="paletteDisplay">
                        </div>
                        
                        <div class="text-center p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-2xl" id="harmonyScore">
                        </div>
                        
                        <div class="text-center mt-4">
                            <button onclick="generatePalette()" 
                                    class="bg-white border-2 border-purple-500 text-purple-600 hover:bg-purple-500 hover:text-white 
                                           font-semibold py-2 px-6 rounded-xl transition-all duration-200">
                                🔄 Создать новую палитру
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                let colorIndex = 1;
                
                function addColor() {
                    if (document.querySelectorAll('.color-input').length >= 9) return;
                    
                    const container = document.getElementById('colorInputs');
                    const addBtn = container.querySelector('button');
                    
                    const colorWrapper = document.createElement('div');
                    colorWrapper.className = 'relative group animate-bounce-in';
                    
                    const colorInput = document.createElement('input');
                    colorInput.type = 'color';
                    colorInput.className = 'color-input w-16 h-16 md:w-20 md:h-20 border-3 border-gray-200 rounded-2xl cursor-pointer hover:scale-110 transition-transform duration-200';
                    colorInput.value = '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0');
                    colorInput.id = 'color' + colorIndex++;
                    
                    const removeBtn = document.createElement('button');
                    removeBtn.innerHTML = '×';
                    removeBtn.className = 'absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-sm opacity-0 group-hover:opacity-100 transition-opacity duration-200 hover:bg-red-600';
                    removeBtn.onclick = () => colorWrapper.remove();
                    
                    colorWrapper.appendChild(colorInput);
                    colorWrapper.appendChild(removeBtn);
                    container.insertBefore(colorWrapper, addBtn);
                }
                
                function clearColors() {
                    const colorInputs = document.querySelectorAll('.color-input');
                    colorInputs.forEach(input => input.parentElement.remove());
                }
                
                async function generatePalette() {
                    const inputs = document.querySelectorAll('.color-input');
                    const inputColors = Array.from(inputs).map(input => {
                        const hex = input.value;
                        return [parseInt(hex.substr(1, 2), 16), parseInt(hex.substr(3, 2), 16), parseInt(hex.substr(5, 2), 16)];
                    });
                    
                    const targetCount = parseInt(document.getElementById('targetCount').value);
                    const randomize = true; // Всегда включена рандомизация
                    
                    document.getElementById('loading').classList.remove('hidden');
                    document.getElementById('results').classList.add('hidden');
                    document.getElementById('error').classList.add('hidden');
                    
                    try {
                        const response = await fetch('/generate', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ 
                                input_colors: inputColors.length > 0 ? inputColors : [], 
                                target_count: targetCount, 
                                randomize: randomize 
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (!result.success) throw new Error(result.error);
                        
                        const display = document.getElementById('paletteDisplay');
                        display.innerHTML = result.palette.map((color, index) => {
                            const hex = '#' + color.map(c => c.toString(16).padStart(2, '0')).join('').toUpperCase();
                            const brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000;
                            const textColor = brightness > 128 ? 'text-gray-800' : 'text-white';
                            
                            return `
                                <div class="palette-color group relative bg-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-200 cursor-pointer transform hover:scale-105 animate-bounce-in" 
                                     style="background-color: rgb(${color[0]}, ${color[1]}, ${color[2]}); animation-delay: ${index * 0.1}s" 
                                     onclick="copyColor('${hex}')" 
                                     title="Нажмите для копирования ${hex}">
                                    <div class="aspect-square flex flex-col items-center justify-center p-4">
                                        <div class="${textColor} font-mono text-sm font-bold mb-2">${hex}</div>
                                        <div class="${textColor} text-xs opacity-80">RGB(${color[0]}, ${color[1]}, ${color[2]})</div>
                                        <div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 rounded-2xl transition-all duration-200 flex items-center justify-center">
                                            <span class="${textColor} opacity-0 group-hover:opacity-100 font-semibold">📋 Копировать</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                        
                        const harmonyLevel = result.harmony_score >= 0.8 ? '🌟 Отлично' : 
                                           result.harmony_score >= 0.6 ? '👍 Хорошо' : 
                                           result.harmony_score >= 0.4 ? '👌 Неплохо' : '🔄 Попробуйте еще';
                        
                        document.getElementById('harmonyScore').innerHTML = `
                            <div class="flex items-center justify-center space-x-4">
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-green-600">${result.harmony_score.toFixed(3)}</div>
                                    <div class="text-sm text-gray-600">Гармоничность</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-xl">${harmonyLevel}</div>
                                    <div class="text-sm text-gray-600">Оценка</div>
                                </div>
                            </div>
                        `;
                        
                        document.getElementById('results').classList.remove('hidden');
                        
                    } catch (error) {
                        document.getElementById('errorText').textContent = error.message;
                        document.getElementById('error').classList.remove('hidden');
                    }
                    
                    document.getElementById('loading').classList.add('hidden');
                }
                
                function copyColor(hex) {
                    navigator.clipboard.writeText(hex).then(() => {
                        // Создаем красивое уведомление
                        const notification = document.createElement('div');
                        notification.innerHTML = `
                            <div class="fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-bounce-in">
                                ✅ Цвет ${hex} скопирован!
                            </div>
                        `;
                        document.body.appendChild(notification);
                        
                        setTimeout(() => {
                            notification.remove();
                        }, 3000);
                    }).catch(() => {
                        alert('Цвет ' + hex + ' скопирован!');
                    });
                }
                
                // Убираем автоматическую генерацию при загрузке
                // window.addEventListener('load', () => {
                //     setTimeout(() => {
                //         generatePalette();
                //     }, 500);
                // });
            </script>
        </body>
        </html>
        """

@app.route('/api')
def api_docs():
    """
    Страница с документацией API
    """
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Colors API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
            .example { background: #e8f4fd; padding: 10px; margin: 10px 0; }
            .color-box { display: inline-block; width: 30px; height: 30px; margin: 2px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h1>🎨 AI Colors - API для генерации цветовых палитр</h1>
        
        <p><a href="/">← Вернуться к веб-интерфейсу</a></p>
        
        <p>Этот API использует нейросеть для генерации гармоничных цветовых палитр на основе входных цветов.</p>
        
        <div class="endpoint">
            <h3>POST /generate</h3>
            <p><strong>Описание:</strong> Генерирует цветовую палитру</p>
            
            <h4>Параметры запроса:</h4>
            <ul>
                <li><strong>input_colors</strong> (список): От 1 до 9 цветов в формате RGB [[R, G, B], ...]</li>
                <li><strong>target_count</strong> (число): Желаемое количество цветов в палитре (2-10)</li>
                <li><strong>randomize</strong> (boolean): Добавить случайность для разнообразия (по умолчанию: true)</li>
            </ul>
            
            <h4>Пример запроса:</h4>
            <div class="example">
                <pre>{
    "input_colors": [[255, 0, 0], [0, 255, 0]],
    "target_count": 5,
    "randomize": true
}</pre>
            </div>
            
            <h4>Пример ответа:</h4>
            <div class="example">
                <pre>{
    "success": true,
    "palette": [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
    "harmony_score": 0.75,
    "input_count": 2,
    "output_count": 5
}</pre>
            </div>
        </div>
        
        <div class="endpoint">
            <h3>GET /health</h3>
            <p><strong>Описание:</strong> Проверка состояния сервиса</p>
        </div>
        
        <div class="endpoint">
            <h3>POST /evaluate</h3>
            <p><strong>Описание:</strong> Оценивает гармоничность существующей палитры</p>
            
            <h4>Параметры запроса:</h4>
            <ul>
                <li><strong>colors</strong> (список): Цвета для оценки в формате RGB</li>
            </ul>
        </div>
        
        <h3>Примеры использования:</h3>
        <div class="example">
            <h4>curl:</h4>
            <pre>curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"input_colors": [[255, 0, 0], [0, 255, 0]], "target_count": 5, "randomize": true}'</pre>
        </div>
        
        <div class="example">
            <h4>Python:</h4>
            <pre>import requests

response = requests.post('http://localhost:5000/generate', json={
    'input_colors': [[255, 0, 0], [0, 255, 0]], 
    'target_count': 5,
    'randomize': True
})
print(response.json())</pre>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Проверка состояния сервиса
    """
    try:
        model_status = "loaded" if model is not None else "not_loaded"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "device": device
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate_palette():
    """
    Генерирует цветовую палитру
    """
    try:
        # Получаем данные запроса
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "Empty JSON request"
            }), 400
        
        # Валидируем входные данные
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
        
        input_colors = data.get('input_colors', [])
        target_count = data['target_count']
        randomize = data.get('randomize', True)  # По умолчанию включена
        
        # Оставляем входные цвета пустыми, если их нет - нейросеть сама сгенерирует базовый цвет
        
        # Конвертируем в кортежи (может быть пустой список)
        input_colors_tuples = [tuple(color) for color in input_colors] if input_colors else []
        
        # Загружаем модель если не загружена
        current_model = load_model()
        
        # Генерируем палитру
        generated_palette = current_model.generate_palette(
            input_colors_tuples, 
            target_count, 
            device,
            randomize=randomize
        )
        
        # Вычисляем оценку гармоничности
        harmony_score = calculate_color_harmony_score(generated_palette)
        
        # Возвращаем результат
        return jsonify({
            "success": True,
            "palette": generated_palette,
            "harmony_score": round(harmony_score, 3),
            "input_count": len(input_colors),
            "output_count": len(generated_palette)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Generation error: {str(e)}"
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_palette():
    """
    Оценивает гармоничность цветовой палитры
    """
    try:
        data = request.get_json()
        
        if not data or 'colors' not in data:
            return jsonify({
                "success": False,
                "error": "Missing field 'colors'"
            }), 400
        
        colors = data['colors']
        
        # Валидируем цвета
        if not isinstance(colors, list) or len(colors) < 2:
            return jsonify({
                "success": False,
                "error": "Colors must contain at least 2 colors"
            }), 400
        
        for i, color in enumerate(colors):
            if not validate_color(color):
                return jsonify({
                    "success": False,
                    "error": f"Invalid color at position {i}: {color}"
                }), 400
        
        # Конвертируем в кортежи и вычисляем гармоничность
        colors_tuples = [tuple(color) for color in colors]
        harmony_score = calculate_color_harmony_score(colors_tuples)
        
        # Harmony classification
        if harmony_score >= 0.8:
            harmony_level = "excellent"
        elif harmony_score >= 0.6:
            harmony_level = "good"
        elif harmony_score >= 0.4:
            harmony_level = "fair"
        else:
            harmony_level = "poor"
        
        return jsonify({
            "success": True,
            "harmony_score": round(harmony_score, 3),
            "harmony_level": harmony_level,
            "colors_count": len(colors)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Evaluation error: {str(e)}"
        }), 500

@app.route('/random', methods=['GET'])
def generate_random_palette():
    """
    Генерирует случайную гармоничную палитру
    """
    try:
        # Случайные параметры
        import random
        
        # Генерируем случайные входные цвета
        num_input = random.randint(1, 3)
        input_colors = []
        for _ in range(num_input):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            input_colors.append((r, g, b))
        
        target_count = random.randint(4, 8)
        
        # Загружаем модель
        current_model = load_model()
        
        # Генерируем палитру
        generated_palette = current_model.generate_palette(
            input_colors,
            target_count,
            device
        )
        
        harmony_score = calculate_color_harmony_score(generated_palette)
        
        return jsonify({
            "success": True,
            "palette": generated_palette,
            "input_colors": input_colors,
            "harmony_score": round(harmony_score, 3),
            "target_count": target_count
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Random palette generation error: {str(e)}"
        }), 500

@app.route('/og.png')
def og_image():
    """
    Обслуживает og.png для социальных сетей
    """
    try:
        return send_from_directory(
            os.path.join(os.path.dirname(__file__), 'templates'),
            'og.png',
            mimetype='image/png'
        )
    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": "Image not found"
        }), 404

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Method not allowed"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Предзагружаем модель при запуске
    print("Запуск AI Colors API...")
    load_model()
    
    # Запускаем сервер
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Отключаем debug в продакшене
    )