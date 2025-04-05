import os
from torchvision import datasets
from PIL import Image

# Создание базовой директории и подпапок
base_dir = './mnist_data'
splits = ['train', 'test', 'valid']

for split in splits:
    for digit in range(10):
        dir_path = os.path.join(base_dir, split, str(digit))
        os.makedirs(dir_path, exist_ok=True)

# Загрузка MNIST датасета
train_dataset = datasets.MNIST(root='data', train=True, download=True)
test_dataset = datasets.MNIST(root='data', train=False, download=True)

# Разделение тренировочных данных на train и valid
train_size = 50000  # 50k примеров для обучения
valid_size = 10000  # 10k примеров для валидации

# Сохранение тренировочных и валидационных данных
for i in range(len(train_dataset)):
    image = train_dataset.data[i]
    label = train_dataset.targets[i].item()
    
    split = 'train' if i < train_size else 'valid'
    filename = os.path.join(base_dir, split, str(label), f"{i}.png")
    
    Image.fromarray(image.numpy(), mode='L').save(filename)

# Сохранение тестовых данных
for i in range(len(test_dataset)):
    image = test_dataset.data[i]
    label = test_dataset.targets[i].item()
    
    filename = os.path.join(base_dir, 'test', str(label), f"test_{i}.png")
    Image.fromarray(image.numpy(), mode='L').save(filename)

print("Данные успешно сохранены в папки:", os.path.abspath(base_dir))