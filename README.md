# Описание проекта
Этот репозиторий содержит реализацию нейронной сети для классификации рукописных цифр из набора данных MNIST. 

Главный файл `script.py` содержит реализацию нейронной сети и ее обучение на наборе данных MNIST. Скрипт имеет два вида работы:
1) обучение: по заданным параметрам модели происходит обучение модели и сохранение ее весов;
2) классификация: для заданного изображения происходит классификация цифры на изображении.

# Установка зависимостей
Убедитесь, что у вас установлен Python 3.7 или выше.
Установите зависимости:

```bash
pip install numpy keras opencv-python
```
# Работа со скриптом
### Обучение
Чтобы обучить модель, запустите скрипт с параметром `--train`:
```bash
python script.py --train
```
Опции:
- `--train`: флаг обучения модели;
- `--path_to_dataset`: путь к датасету для обучения;
- `--learning_rate`: скорость обучения (*default=0.001*);
- `--epochs`: количество эпох обучения (*default=50*);
- `--hidden_sizes`: массив, который отвечает за количество нейронов в скрытых слоях, задается в виде `--hidden_sizes "[]"` (*default=[]*);
- `--weights_path`: путь куда сохранять веса модели (*default=weights.npz*).

Пример запуска с параметрами:
```bash
python script.py --train --path_to_dataset "mnist_data" --learning_rate "0.1" --hidden_size "[1024]" --epochs 50
```
Для текущего скрипта вывод будет следующий:
```bash
$ python script.py --train --path_to_dataset "mnist_data" --learning_rate "0.1" --hidden_size "[1024]" --epochs 50
Loading dataset from mnist_data
loading train data...
loading test data...
loading validation data...
Dataset loaded:
train size: 50000
test size: 10000
validation size: 10000
Begin training...
Epoch 1/50 | Loss: 0.2423 | Train Acc: 0.0725 | Val Acc: 0.2175
Epoch 2/50 | Loss: 0.2187 | Train Acc: 0.2113 | Val Acc: 0.4052
Epoch 3/50 | Loss: 0.2030 | Train Acc: 0.3985 | Val Acc: 0.5419
Epoch 4/50 | Loss: 0.1892 | Train Acc: 0.5323 | Val Acc: 0.6261
Epoch 5/50 | Loss: 0.1767 | Train Acc: 0.6097 | Val Acc: 0.6770
Epoch 6/50 | Loss: 0.1655 | Train Acc: 0.6590 | Val Acc: 0.7128
Epoch 7/50 | Loss: 0.1555 | Train Acc: 0.6928 | Val Acc: 0.7355
Epoch 8/50 | Loss: 0.1466 | Train Acc: 0.7174 | Val Acc: 0.7547
Epoch 9/50 | Loss: 0.1387 | Train Acc: 0.7351 | Val Acc: 0.7697
Epoch 10/50 | Loss: 0.1316 | Train Acc: 0.7493 | Val Acc: 0.7817
Epoch 11/50 | Loss: 0.1252 | Train Acc: 0.7605 | Val Acc: 0.7909
Epoch 12/50 | Loss: 0.1195 | Train Acc: 0.7707 | Val Acc: 0.7998
Epoch 13/50 | Loss: 0.1143 | Train Acc: 0.7790 | Val Acc: 0.8073
Epoch 14/50 | Loss: 0.1097 | Train Acc: 0.7856 | Val Acc: 0.8134
Epoch 15/50 | Loss: 0.1055 | Train Acc: 0.7920 | Val Acc: 0.8198
Epoch 16/50 | Loss: 0.1017 | Train Acc: 0.7979 | Val Acc: 0.8264
Epoch 17/50 | Loss: 0.0983 | Train Acc: 0.8036 | Val Acc: 0.8309
Epoch 18/50 | Loss: 0.0951 | Train Acc: 0.8081 | Val Acc: 0.8353
Epoch 19/50 | Loss: 0.0922 | Train Acc: 0.8125 | Val Acc: 0.8384
Epoch 20/50 | Loss: 0.0896 | Train Acc: 0.8168 | Val Acc: 0.8421
Epoch 21/50 | Loss: 0.0871 | Train Acc: 0.8206 | Val Acc: 0.8441
Epoch 22/50 | Loss: 0.0849 | Train Acc: 0.8240 | Val Acc: 0.8480
Epoch 23/50 | Loss: 0.0828 | Train Acc: 0.8267 | Val Acc: 0.8504
Epoch 24/50 | Loss: 0.0808 | Train Acc: 0.8299 | Val Acc: 0.8527
Epoch 25/50 | Loss: 0.0790 | Train Acc: 0.8321 | Val Acc: 0.8544
Epoch 26/50 | Loss: 0.0773 | Train Acc: 0.8346 | Val Acc: 0.8558
Epoch 27/50 | Loss: 0.0757 | Train Acc: 0.8368 | Val Acc: 0.8577
Epoch 28/50 | Loss: 0.0743 | Train Acc: 0.8391 | Val Acc: 0.8595
Epoch 29/50 | Loss: 0.0729 | Train Acc: 0.8405 | Val Acc: 0.8607
Epoch 30/50 | Loss: 0.0715 | Train Acc: 0.8427 | Val Acc: 0.8623
Epoch 31/50 | Loss: 0.0703 | Train Acc: 0.8449 | Val Acc: 0.8640
Epoch 32/50 | Loss: 0.0691 | Train Acc: 0.8464 | Val Acc: 0.8657
Epoch 33/50 | Loss: 0.0680 | Train Acc: 0.8486 | Val Acc: 0.8668
Epoch 34/50 | Loss: 0.0670 | Train Acc: 0.8502 | Val Acc: 0.8685
Epoch 35/50 | Loss: 0.0660 | Train Acc: 0.8516 | Val Acc: 0.8701
Epoch 36/50 | Loss: 0.0651 | Train Acc: 0.8529 | Val Acc: 0.8720
Epoch 37/50 | Loss: 0.0642 | Train Acc: 0.8544 | Val Acc: 0.8729
Epoch 38/50 | Loss: 0.0633 | Train Acc: 0.8555 | Val Acc: 0.8744
Epoch 39/50 | Loss: 0.0625 | Train Acc: 0.8567 | Val Acc: 0.8758
Epoch 40/50 | Loss: 0.0617 | Train Acc: 0.8579 | Val Acc: 0.8773
Epoch 41/50 | Loss: 0.0610 | Train Acc: 0.8593 | Val Acc: 0.8783
Epoch 42/50 | Loss: 0.0602 | Train Acc: 0.8602 | Val Acc: 0.8793
Epoch 43/50 | Loss: 0.0596 | Train Acc: 0.8612 | Val Acc: 0.8800
Epoch 44/50 | Loss: 0.0589 | Train Acc: 0.8624 | Val Acc: 0.8809
Epoch 45/50 | Loss: 0.0583 | Train Acc: 0.8631 | Val Acc: 0.8816
Epoch 46/50 | Loss: 0.0577 | Train Acc: 0.8642 | Val Acc: 0.8822
Epoch 47/50 | Loss: 0.0571 | Train Acc: 0.8652 | Val Acc: 0.8832
Epoch 48/50 | Loss: 0.0565 | Train Acc: 0.8659 | Val Acc: 0.8841
Epoch 49/50 | Loss: 0.0560 | Train Acc: 0.8671 | Val Acc: 0.8851
Epoch 50/50 | Loss: 0.0555 | Train Acc: 0.8680 | Val Acc: 0.8854
Test accuracy: 0.8780
Test precision: 0.8770
Test recall: 0.8759
Test F1-score: 0.8758
Confusion Matrix:
[[ 941    0    4    3    0   12   13    1    6    0]
 [   0 1104    3    4    0    1    4    0   19    0]
 [  22   15  857   23   19    2   23   21   45    5]
 [   6    2   23  891    1   26    8   19   24   10]
 [   3    8    4    1  870    0   20    2    7   67]
 [  18   13    6   63   19  682   20   10   38   23]
 [  17    3   12    2   17   20  880    2    5    0]
 [   4   32   26    2   13    1    0  899   10   41]
 [  10   15   11   43   12   33   18   16  796   20]
 [  15   14    7   14   46    9    1   33   10  860]]
```
### Требования к обучающему датасету 
Датасет должен быть разделен на три папки: `train`, `test` и `valid`
Каждая папка должна содержать папки с именами классов, в которых находятся изображения классов.
В каждой папке должны быть файлы изображений в формате PNG или JPEG.

Папки `test` и `valid` являются опциональными
Наименования файлов изображений не имеют значения.

Пример иерархии:
```
mnist_data
         ├───test
         │   ├───0
         │   │   ├───"file.jpg"
         │   │   └── ...
         │   ├───1
         │   ...
         │   └───9
         ├───train
         │   ├───0
         │   │   ├───"file.jpg"
         │   │   └── ...
         │   ├───1
         │   ...
         │   └───9
         └───valid
             ├───0
             │   ├───"file.jpg"
             │   └── ...
             ├───1
             ...
             └───9
```

В репозитории есть скрипт для генерации датасета - `generate_dataset.py`
Данный скрипт генерирует в текущей директории датасет, который уже разделен на `train`, `test` и `valid`

Пропишите следующий скрипт для генерации:
```bash
python generate_dataset.py
```

### Классификация изображений
Запустите скрипт, указав путь к изображению:
```bash
python script.py path/to/your/image.png
```

Опции:
* `--probs`: Вывести вероятности для всех классов (по умолчанию выводится только предсказанный класс).
* `--weights_path`: путь от куда загружать веса модели (*default=weights.npz*).

### **Требования к изображению**:
* Формат: PNG, JPG, BMP.
* Рекомендуемый размер: 28x28 пикселей (скрипт автоматически изменит размер).

### **Пример работы**:
Вывод классифицированной цифры:
```bash
$ python script.py ./5.png
Predicted digit: 5
```

Вывод вероятностей:
```bash
$ python script.py ./1.jpg --probs
Class probabilities:
0: 0.0034
1: 0.8317
2: 0.0344
3: 0.0347
4: 0.0105
5: 0.0192
6: 0.0184
7: 0.0146
8: 0.0228
9: 0.0102
```

