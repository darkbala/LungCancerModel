import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset', 'Datasets')

# Проверка существования пути
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")

print(f"Dataset path is: {DATASET_PATH}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = ImageFolder(DATASET_PATH, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Создание модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 класса
model = model.to(device)

# Настройка обучения
epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader)}")

# Сохранение модели
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '..','path_to_model', 'model.pth')
torch.save(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
