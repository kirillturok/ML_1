import os
import cv2
import random
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

DATASET_FOLDER = r'./dataset'
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
DATA_COLUMN = 'data'
LABELS_COLUMN = 'labels'
HASHED_DATA_COLUMN = 'data_bytes'
MAX_ITERATIONS = 1_000
TRAIN_SIZES = [50, 100, 1_000, 50_000]


def show_images():
    def show_one_image(image_folder):
        file = random.choice(os.listdir(image_folder))
        image_path = os.path.join(image_folder, file)
        img = cv2.imread(image_path)
        pyplot.imshow(img)
        pyplot.show()

    for class_item in CLASSES:
        image_folder = os.path.join(DATASET_FOLDER, class_item)
        show_one_image(image_folder)


def get_class_data(folder_path):
    result_data = list()
    files = os.listdir(folder_path)
    for file in files:
        image_path = os.path.join(folder_path, file)
        img = cv2.imread(image_path)
        if img is not None:
            result_data.append(img.reshape(-1))

    return result_data


def create_data_frame():
    data = list()
    labels = list()
    for class_item in CLASSES:
        class_folder_path = os.path.join(DATASET_FOLDER, class_item)
        class_data = get_class_data(class_folder_path)

        data.extend(class_data)
        labels.extend([CLASSES.index(class_item) for _ in range(len(class_data))])

    data_frame = pandas.DataFrame({DATA_COLUMN: data, LABELS_COLUMN: labels})

    return data_frame


def remove_duplicates(data):
    data_bytes = [item.tobytes() for item in data[DATA_COLUMN]]
    data[HASHED_DATA_COLUMN] = data_bytes
    data.sort_values(HASHED_DATA_COLUMN, inplace=True)
    data.drop_duplicates(subset=HASHED_DATA_COLUMN, keep='first', inplace=True)
    data.pop(HASHED_DATA_COLUMN)

    return data


def shuffle_data(data):
    data_shuffled = data.sample(frac=1, random_state=42)

    return data_shuffled


def get_images_counts(data_frame):
    classes_images_counts = list()
    for class_index in range(len(CLASSES)):
        labels = data_frame[LABELS_COLUMN]
        class_rows = data_frame[labels == class_index]
        class_count = len(class_rows)

        classes_images_counts.append(class_count)

    return classes_images_counts


def check_balance(data_frame):
    classes_images_counts = get_images_counts(data_frame)

    max_images_count = max(classes_images_counts)
    avg_images_count = sum(classes_images_counts) / len(classes_images_counts)
    balance_percent = avg_images_count / max_images_count

    if balance_percent > 0.85:
        print("Classes are balanced")
    else:
        print("Classes are not balanced")


def divide_into_subsamples(data_frame):
    data = numpy.array(list(data_frame[DATA_COLUMN].values), numpy.float32)
    labels = numpy.array(list(data_frame[LABELS_COLUMN].values), numpy.float32)

    x_train, x_remaining, y_train, y_remaining = train_test_split(data, labels, train_size=0.4)
    x_valid, x_test, y_valid, y_test = train_test_split(x_remaining, y_remaining, test_size=0.04)

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def get_logistic_regression(x_train, y_train, x_test, y_test):
    test_scores = list()
    logistic_regression = LogisticRegression(max_iter=MAX_ITERATIONS)
    for train_size in TRAIN_SIZES:
        logistic_regression.fit(x_train[:train_size], y_train[:train_size])

        score = logistic_regression.score(x_test, y_test)
        test_scores.append(score)

    return test_scores


def show_result_plot(test_scores):
    pyplot.figure()
    pyplot.xlabel('Training data size')
    pyplot.ylabel('Accuracy')
    pyplot.grid()

    ticks = range(len(TRAIN_SIZES))
    pyplot.xticks(ticks, TRAIN_SIZES)
    pyplot.plot(ticks, test_scores, 'o-', color='g', label='Testing score')

    pyplot.show()


def main():
    #show_images()

    data_frame = create_data_frame()
    data_frame = remove_duplicates(data_frame)
    check_balance(data_frame)
    data_frame = shuffle_data(data_frame)

    x_train, y_train, x_test, y_test, *_ = divide_into_subsamples(data_frame)

    test_scores = get_logistic_regression(x_train, y_train, x_test, y_test)
    show_result_plot(test_scores)


main()
