## Шумоподавление на мел-спектрограммах

В данном проекте решаются две задачи:
1. Классификация  мел-спектрограмм на зашумленные и чистые
2. Шумоподавление на мел-спектрограммах

В решении используется архитектура UNet c residual блоками. Для задачи классификации к выходу энкодера добавляется полносвязный слой, для задачи шумоподавления в качестве выхода используется выход декодера.


### Pipeline

1. Для запуска обучения нужно выполнить следующий скрипт:
    
    ```bash
   python3 train.py -t ../train -v ../validation
    ```
   В качестве аргумента -t необходимо передать путь к директории с тренировочной выборкой, а -v - путь к директории с 
валидационной выборкой. 
   Логи тренировки будут сохранены в папку ./logs, а веса модели в папку ./logs/weights 
    
2. Для запуска тестирования модели нужно выполнить следующий скрипт:
    
    ```bash
    python3 test.py -t ../val/ -w ./logs/pretrained_weights/model_weights
    ```
   В качестве аргумента -t необходимо передать путь к директории с тестовой выборкой, а в аргумент -w путь к весам модели.
   
3. Для запуска модели на пользовательских .npy файлах используйте:

    ```bash
   # for classification task
    python3 predict.py -m classification -f ../train/noisy/20/20_205_20-205-0004.npy -w logs/pretrained_weights/model_weights
   
    # for denoising task
    python3 predict.py -m denoising -f ../train/noisy/20/20_205_20-205-0004.npy -w logs/pretrained_weights/model_weights -o output/
    ```
 В качестве аргумента -m необходимо передать classification или denoising в зависимости от решаемой задачи. 
 -f - путь к npy файлу, -w путь к весам модели. Для задачи шумоподавления нужно передать путь к выходным файлам в аргумент -o. 
 В данной папке будут созданы файл out.npy и картинка mel.png с результатами обработки. Результатом выполнения задачи классификации будет вывод строки ‘noisy’ или ‘clean’.

   