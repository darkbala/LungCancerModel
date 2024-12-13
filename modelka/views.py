from django.shortcuts import render
from django.http import JsonResponse
from .models import LungScan
from .forms import LungScanForm
from torchvision import transforms
from PIL import Image
import os

# Путь к модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'path_to_model', 'model.pth')  # Укажите правильный путь к модели

# Проверяем, существует ли файл модели
if os.path.exists(MODEL_PATH):
    import torch
    # Загрузка обученной модели
    model = torch.load(MODEL_PATH)
    model.eval()
else:
    # Если модели нет, используем заглушку
    print("Warning: Model not found. Using a mock function for predictions.")
    model = None

def predict_image(image_path):
    """
    Функция предсказания. Возвращает результат анализа снимка.
    Если модель отсутствует, возвращает случайный результат.
    """
    if model:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Приводим изображение к стандартному размеру
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = Image.open(image_path).convert('RGB')  # Конвертация изображения в формат RGB
        image = transform(image).unsqueeze(0)  # Добавляем batch размерности
        with torch.no_grad():
            output = model(image)  # Прогнозируем
        return torch.argmax(output).item()  # Возвращаем метку класса (0 или 1)
    else:
        # Возвращаем случайный результат (заглушка)
        import random
        return random.choice([0, 1])  # 0 - Benign, 1 - Malignant

def upload_scan(request):
    """
    Обработчик загрузки снимков. Возвращает результат анализа.
    """
    if request.method == 'POST':
        form = LungScanForm(request.POST, request.FILES)
        if form.is_valid():
            scan = form.save()  # Сохраняем загруженный снимок
            # Предсказание результата
            result = predict_image(scan.image.path)
            scan.result = "Malignant" if result == 1 else "Benign"
            scan.save()  # Сохраняем результат в базе данных
            return JsonResponse({'result': scan.result})  # Возвращаем JSON-ответ с результатом
    else:
        form = LungScanForm()
    return render(request, 'home.html', {'form': form})
