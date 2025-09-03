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
    Валидирует входные данные
    """
    # Проверяем наличие обязательных полей
    if 'input_colors' not in data:
        return False, "Отсутствует поле 'input_colors'"
    
    if 'target_count' not in data:
        return False, "Отсутствует поле 'target_count'"
    
    input_colors = data['input_colors']
    target_count = data['target_count']
    
    # Проверяем входные цвета
    if not isinstance(input_colors, list) or len(input_colors) == 0:
        return False, "input_colors должен быть непустым списком"
    
    if len(input_colors) > 9:
        return False, "Максимум 9 входных цветов"
    
    for i, color in enumerate(input_colors):
        if not validate_color(color):
            return False, f"Некорректный цвет в позиции {i}: {color}"
    
    # Проверяем целевое количество
    if not isinstance(target_count, int) or target_count < 2 or target_count > 10:
        return False, "target_count должен быть целым числом от 2 до 10"
    
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
            <title>🎨 AI Colors - Генератор цветовых палитр</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 15px; padding: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
                h1 { color: #333; text-align: center; margin-bottom: 30px; }
                .color-inputs { display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }
                .color-input { width: 80px; height: 80px; border: 3px solid #ddd; border-radius: 50%; cursor: pointer; }
                .controls { display: flex; gap: 20px; align-items: end; flex-wrap: wrap; }
                .control-group { display: flex; flex-direction: column; }
                .control-group label { margin-bottom: 5px; font-weight: 500; }
                .control-group input, select { padding: 10px; border: 2px solid #ddd; border-radius: 8px; font-size: 16px; }
                .generate-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 25px; font-size: 16px; cursor: pointer; }
                .palette-display { display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; justify-content: center; }
                .palette-color { width: 100px; height: 100px; border-radius: 15px; display: flex; align-items: end; padding: 10px; color: white; font-family: monospace; font-size: 12px; cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.2); }
                .results { display: none; margin-top: 30px; }
                .loading { display: none; text-align: center; color: #667eea; }
                .error { background: #fff2f0; border: 1px solid #ffccc7; color: #cf1322; padding: 15px; border-radius: 8px; margin: 20px 0; display: none; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎨 AI Colors</h1>
                <p>Генератор гармоничных цветовых палитр на основе нейросетей</p>
                
                <div class="color-inputs">
                    <input type="color" class="color-input" value="#ff0000" id="color1">
                    <input type="color" class="color-input" value="#00ff00" id="color2">
                    <button onclick="addColor()" style="width: 80px; height: 80px; border: 3px dashed #ddd; border-radius: 50%; background: transparent; cursor: pointer; font-size: 2em; color: #ddd;">+</button>
                </div>
                
                <div class="controls">
                    <div class="control-group">
                        <label>Количество цветов:</label>
                        <input type="number" id="targetCount" min="2" max="10" value="5">
                    </div>
                    <div class="control-group">
                        <label>Рандомизация:</label>
                        <select id="randomize">
                            <option value="true">Включена</option>
                            <option value="false">Выключена</option>
                        </select>
                    </div>
                    <button class="generate-btn" onclick="generatePalette()">✨ Генерировать</button>
                </div>
                
                <div class="loading" id="loading">
                    <p>Генерируем палитру...</p>
                </div>
                
                <div class="error" id="error"></div>
                
                <div class="results" id="results">
                    <h3>🌈 Результат:</h3>
                    <div class="palette-display" id="paletteDisplay"></div>
                    <div id="harmonyScore"></div>
                </div>
            </div>
            
            <script>
                let colorIndex = 3;
                
                function addColor() {
                    if (document.querySelectorAll('.color-input').length >= 9) return;
                    const container = document.querySelector('.color-inputs');
                    const addBtn = container.querySelector('button');
                    const newInput = document.createElement('input');
                    newInput.type = 'color';
                    newInput.className = 'color-input';
                    newInput.value = '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0');
                    newInput.id = 'color' + colorIndex++;
                    container.insertBefore(newInput, addBtn);
                }
                
                async function generatePalette() {
                    const inputs = document.querySelectorAll('.color-input');
                    const inputColors = Array.from(inputs).map(input => {
                        const hex = input.value;
                        return [parseInt(hex.substr(1, 2), 16), parseInt(hex.substr(3, 2), 16), parseInt(hex.substr(5, 2), 16)];
                    });
                    
                    const targetCount = parseInt(document.getElementById('targetCount').value);
                    const randomize = document.getElementById('randomize').value === 'true';
                    
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('results').style.display = 'none';
                    document.getElementById('error').style.display = 'none';
                    
                    try {
                        const response = await fetch('/generate', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ input_colors: inputColors, target_count: targetCount, randomize: randomize })
                        });
                        
                        const result = await response.json();
                        
                        if (!result.success) throw new Error(result.error);
                        
                        const display = document.getElementById('paletteDisplay');
                        display.innerHTML = result.palette.map(color => 
                            `<div class="palette-color" style="background-color: rgb(${color[0]}, ${color[1]}, ${color[2]})" 
                                  onclick="copyColor('#${color.map(c => c.toString(16).padStart(2, '0')).join('')}')" 
                                  title="Нажмите для копирования">
                                #${color.map(c => c.toString(16).padStart(2, '0').toUpperCase()).join('')}
                            </div>`
                        ).join('');
                        
                        document.getElementById('harmonyScore').innerHTML = 
                            `<p>🎵 Оценка гармоничности: ${result.harmony_score.toFixed(3)}/1.000</p>`;
                        
                        document.getElementById('results').style.display = 'block';
                        
                    } catch (error) {
                        document.getElementById('error').innerHTML = '❌ Ошибка: ' + error.message;
                        document.getElementById('error').style.display = 'block';
                    }
                    
                    document.getElementById('loading').style.display = 'none';
                }
                
                function copyColor(hex) {
                    navigator.clipboard.writeText(hex).then(() => {
                        alert('Цвет ' + hex + ' скопирован!');
                    });
                }
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
                "error": "Пустой JSON"
            }), 400
        
        # Валидируем входные данные
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": error_msg
            }), 400
        
        input_colors = data['input_colors']
        target_count = data['target_count']
        randomize = data.get('randomize', True)  # По умолчанию включена
        
        # Конвертируем в кортежи
        input_colors_tuples = [tuple(color) for color in input_colors]
        
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
            "error": f"Ошибка генерации: {str(e)}"
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
                "error": "Отсутствует поле 'colors'"
            }), 400
        
        colors = data['colors']
        
        # Валидируем цвета
        if not isinstance(colors, list) or len(colors) < 2:
            return jsonify({
                "success": False,
                "error": "colors должен содержать минимум 2 цвета"
            }), 400
        
        for i, color in enumerate(colors):
            if not validate_color(color):
                return jsonify({
                    "success": False,
                    "error": f"Некорректный цвет в позиции {i}: {color}"
                }), 400
        
        # Конвертируем в кортежи и вычисляем гармоничность
        colors_tuples = [tuple(color) for color in colors]
        harmony_score = calculate_color_harmony_score(colors_tuples)
        
        # Классификация гармоничности
        if harmony_score >= 0.8:
            harmony_level = "отлично"
        elif harmony_score >= 0.6:
            harmony_level = "хорошо"
        elif harmony_score >= 0.4:
            harmony_level = "удовлетворительно"
        else:
            harmony_level = "плохо"
        
        return jsonify({
            "success": True,
            "harmony_score": round(harmony_score, 3),
            "harmony_level": harmony_level,
            "colors_count": len(colors)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Ошибка оценки: {str(e)}"
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
            "error": f"Ошибка генерации случайной палитры: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint не найден"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "success": False,
        "error": "Метод не разрешен"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Внутренняя ошибка сервера"
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