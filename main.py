import numpy as np
import cv2
import argparse

WEIGHTS_FILE = "model_weights.npz"

data = np.load(WEIGHTS_FILE)
layers = data['arr_0']
W = []
b = []

for i in range(0, layers):
    _w = data[f"arr_{i + 1}"]
    _b = data[f"arr_{layers + i + 1}"]

    W.append(_w)
    b.append(_b)

def predict(X: np.array, probabilities=True):
    for i in range(0, layers):
        X = X @ W[i] + b[i]
    a2 = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    if probabilities:
        return a2
    
    return np.argmax(a2)

def preprocess_image(image_path):
    # Загрузка изображения в градациях серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Ресайз до 28x28 с интерполяцией
    img = cv2.resize(img, (28, 28))
    
    # Нормализация и преобразование в вектор
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 784)  # Преобразуем в 1x784

    
    return img

if __name__ == "__main__":
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description='MNIST Digit Classifier')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--probs', action='store_true', 
                       help='Show probabilities for all classes')
    args = parser.parse_args()

    try:
        # Предобработка изображения
        processed_img = preprocess_image(args.image_path)
        
        # Предсказание
        if args.probs:
            probabilities = predict(processed_img)
            print("Class probabilities:")
            for i, prob in enumerate(probabilities[0]):
                print(f"{i}: {prob:.4f}")
        else:
            prediction = predict(processed_img, probabilities=False)
            print(f"Predicted digit: {prediction}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
            
