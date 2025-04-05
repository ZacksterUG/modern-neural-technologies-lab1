import os
import numpy as np
from PIL import Image
import cv2
import argparse
import ast
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def process_directory(path):
    """Обрабатывает изображения в указанной директории и возвращает данные и метки."""
    if path is None:
        print('Warning: No path provided')
        return None, None
    
    X = []
    y = []
    
    for label in range(10):
        dir_path = os.path.join(path, str(label))
        if not os.path.isdir(dir_path):
            print('Warning: no directory for label', label)
            continue
        
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                with Image.open(file_path) as img:
                    # Конвертация в градации серого и изменение размера
                    img_gray = img.convert('L').resize((28, 28), Image.LANCZOS)
                    img_array = np.array(img_gray)
                    
                    # Определение необходимости инверсии цветов
                    mean_brightness = np.mean(img_array)
                    if mean_brightness > 127:  # Если преобладает светлый фон
                        img_array = 255 - img_array  # Инверсия цветов
                    
                    # Нормализация и линеаризация
                    X.append((img_array / 255.0).flatten())
                    y.append(to_categorical(label, 10))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return (np.array(X), np.array(y)) if X else (None, None)

def prepare_mnist_data(train_path, test_path=None, valid_path=None):
    """
    Подготавливает данные в формате MNIST с автоматической коррекцией цветов.
    
    Аргументы:
    train_path  - путь к тренировочным данным
    test_path   - путь к тестовым данным (опционально)
    valid_path  - путь к валидационным данным (опционально)
    
    Возвращает:
    Кортеж (X_train, y_train, X_test, y_test, X_valid, y_valid)
    """
    print('loading train data...')
    X_train, y_train = process_directory(train_path)
    print('loading test data...')
    X_test, y_test = process_directory(test_path)
    print('loading validation data...')
    X_valid, y_valid = process_directory(valid_path)
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

def train_model(X, y, X_val, y_val, 
                hidden_size=[512], 
                learning_rate=0.001, epochs=10):
    input_size = X.shape[1]
    output_size = y.shape[1]
    samples = X.shape[0]
    hidden_sizes = [*hidden_size, output_size]
    layers = len(hidden_sizes)
    W = []
    b = []
    np.random.seed(42)
    for i in range(0, layers):
        # Регуляризация Ксавьера
        W_i = np.random.randn(input_size     if i == 0          else hidden_size[i - 1], 
                              hidden_size[i] if i != layers - 1 else output_size) 
        W_i *= np.sqrt(2./(input_size if i == 0 else hidden_size[i - 1]))

        b_i = np.zeros(hidden_size[i] if i != layers - 1 else output_size)

        W.append(W_i)
        b.append(b_i)

    # Функции активации
    softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    relu = lambda x: np.maximum(x, 0)
    relu_derivative = lambda x: np.where(x > 0, 1, 0)
    history = {
        'loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        # Перемешивание данных
        indices = np.random.permutation(samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Прямое распространение
        h = [X_shuffled]
        z_list = []
        for i in range(layers):
            z = h[i] @ W[i] + b[i]
            z_list.append(z)
            h.append(relu(z) if i < layers-1 else z) 

        # Вычисление потерь
        y_pred = softmax(z_list[-1])
        loss = -np.mean(y_shuffled * np.log(y_pred + 1e-8))

        # Обратное распространение
        dl = (y_pred - y_shuffled) / samples

        for i in reversed(range(layers)):
            if i != layers - 1: dl *= relu_derivative(z_list[i])
            
            # Вычисление градиентов
            dW = h[i].T @ dl
            db = np.sum(dl, axis=0)
            
            # Обновление весов и смещений
            W[i] -= learning_rate * dW
            b[i] -= learning_rate * db
            
            # Передача градиента предыдущему слою
            if i > 0: dl = dl @ W[i].T

        # Вычисление метрик
        train_preds = np.argmax(y_pred, axis=1)
        train_labels = np.argmax(y_shuffled, axis=1)
        train_acc = np.mean(train_preds == train_labels)
        
        # Валидационная точность

        if X_val is not None: 
            z_val = X_val

            for i in range(layers):
                z_val = z_val @ W[i] + b[i]

                if i != layers - 1:
                    z_val = relu(z_val)
            y_val_pred = softmax(z_val)
            val_preds = np.argmax(y_val_pred, axis=1)
            val_labels = np.argmax(y_val, axis=1)
            val_acc = np.mean(val_preds == val_labels)

        history['loss'].append(loss)
        history['accuracy'].append(train_acc)

        print(f'Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f}' + 
             (f' | Val Acc: {val_acc:.4f}' if X_val is not None else ''))
        

    return W, b, layers, history

def train_by_dataset(path_to_dataset, epochs, learning_rate, layers, weights_path):
    # Загрузка данных
    if not os.path.exists(os.path.join(path_to_dataset, 'train')):
        raise Exception('Train dataset must exist')
    
    train_path = os.path.join(path_to_dataset, 'train')
    test_path = os.path.join(path_to_dataset, 'test')
    val_path = os.path.join(path_to_dataset, 'valid')

    print(f"Loading dataset from {path_to_dataset}")
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_mnist_data(train_path, 
                                                                        test_path if os.path.exists(os.path.join(path_to_dataset, 'train')) else None, 
                                                                        val_path  if os.path.exists(os.path.join(path_to_dataset, 'valid')) else None)
    print(f"Dataset loaded:")
    print(f"train size: {X_train.shape[0]}")

    if X_test is not None:
        print(f"test size: {X_test.shape[0]}")
    else:
        print("test data not provided")

    if X_val is not None:
        print(f"validation size: {X_val.shape[0]}")
    else:
        print("validation data not provided")

    # Обучение
    print("Begin training...")
    W, b, layers, history = train_model(X_train, y_train, X_val, y_val, layers, learning_rate, epochs)
    
    # Сохранение весов
    np.savez(weights_path, layers, *W, *b)
    
    if X_test is not None:
        y_pred_probs = predict(X_test, layers, W, b, probabilities=True)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Вычисление метрик
        test_acc = np.mean(y_pred_labels == y_test_labels)
        test_prec = precision_score(y_test_labels, y_pred_labels, average='macro')
        test_rec = recall_score(y_test_labels, y_pred_labels, average='macro')
        test_f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

        # Вывод метрик
        print(f'Test accuracy: {test_acc:.4f}')
        print(f'Test precision: {test_prec:.4f}')
        print(f'Test recall: {test_rec:.4f}')
        print(f'Test F1-score: {test_f1:.4f}')

        # Вывод confusion matrix
        conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
        print('Confusion Matrix:')
        print(conf_matrix)
  
def parse_args():
    parser = argparse.ArgumentParser(description='Обработчик аргументов для обучения и предсказания.')
    
    # Основные флаги режимов
    parser.add_argument('--train', action='store_true', help='Активировать режим обучения')
    parser.add_argument('--probs', action='store_true', help='Выводить вероятности вместо меток классов')
    
    # Общий аргумент для путей к весам
    parser.add_argument('--weights_path', type=str, default='weights.npz',
                       help='Путь для сохранения/загрузки весов (по умолчанию: weights.npz)')
    
    # Аргументы для обучения
    parser.add_argument('--path_to_dataset', help='Обязательный путь к обучающему датасету')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Скорость обучения (по умолчанию: 0.001)')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох (по умолчанию: 50)')
    parser.add_argument('--hidden_sizes', type=str, default='[]',
                       help='Список размеров скрытых слоев (по умолчанию: "[]")')
    
    # Аргумент для предсказания
    parser.add_argument('path', nargs='?', help='Обязательный путь к данным для предсказания')
    
    args = parser.parse_args()
    
    # Проверка режима обучения
    if args.train:
        if not args.path_to_dataset:
            parser.error('Для режима обучения необходимо указать --path_to_dataset')
        if args.path:
            parser.error('Аргумент path недопустим в режиме обучения')
    else:
        if not args.path:
            parser.error('Для режима предсказания необходимо указать путь к данным')
    
    if args.train and args.probs:
        parser.error('Флаг --probs недопустим в режиме обучения')
    
    if not args.train:
        training_args = {
            'path_to_dataset': args.path_to_dataset,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'hidden_sizes': args.hidden_sizes,
        }
        defaults = {
            'path_to_dataset': None,
            'learning_rate': 0.001,
            'epochs': 50,
            'hidden_sizes': '[]',
        }
        for arg in training_args:
            if training_args[arg] != defaults[arg]:
                parser.error(f'Аргумент --{arg} недопустим в режиме предсказания')
    
    # Парсинг hidden_sizes
    try:
        hidden_sizes = ast.literal_eval(args.hidden_sizes)
        if not isinstance(hidden_sizes, list) or not all(isinstance(x, int) for x in hidden_sizes):
            raise ValueError
        args.hidden_sizes = hidden_sizes
    except:
        parser.error('Неверный формат hidden_sizes. Используйте список целых чисел в кавычках, например "[64, 32]"')
    
    return args

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

def predict(X: np.array, layers, W, b, probabilities=True):
    for i in range(0, layers):
        X = X @ W[i] + b[i]
        if i != layers - 1:
            X = np.maximum(X, 0)
    a2 = np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    if probabilities:
        return a2
    
    return np.argmax(a2)

def predict_by_path(path, weights_path, probs):
    data = np.load(weights_path)
    layers = data['arr_0']

    W = []
    b = []
    
    for i in range(0, layers):
        _w = data[f"arr_{i + 1}"]
        _b = data[f"arr_{layers + i + 1}"]
    
        W.append(_w)
        b.append(_b)

    processed_img = preprocess_image(path)

    if probs:
        probabilities = predict(processed_img, layers, W, b)
        print("Class probabilities:")
        for i, prob in enumerate(probabilities[0]):
            print(f"{i}: {prob:.4f}")
    else:
        prediction = predict(processed_img, layers, W, b, probabilities=False)
        print(f"Predicted digit: {prediction}")

if __name__ == '__main__':
    args = parse_args()
    
    if not args.train:
        path = args.path
        weights_path = args.weights_path
        probs = args.probs

        predict_by_path(path, weights_path, probs)
    else:
        path_to_dataset = args.path_to_dataset
        epochs = args.epochs
        learning_rate = args.learning_rate
        hidden_sizes = args.hidden_sizes
        weights_path = args.weights_path

        train_by_dataset(path_to_dataset, epochs, learning_rate, hidden_sizes, weights_path)