import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(train_dataset, test_dataset), ds_info = tfds.load(
    'bee_dataset',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=False,
    with_info=True
)


def label_to_string(label):
    label_str = []
    for key, value in label.items():
        if value.numpy() == 1.0:
            label_str.append(key.replace('_output', ''))
    return ', '.join(label_str) if label_str else 'No activity'


def plot_images(dataset, num_images=5):
    plt.figure(figsize=(15, 15))
    for i, example in enumerate(dataset.take(num_images)):
        ax = plt.subplot(3, 3, i + 1)
        image = example['input']  
        label = example['output']  
        label_text = label_to_string(label)  
        plt.imshow(image)
        plt.title(f"Activity: {label_text}")
        plt.axis("off")
    plt.show()


print("Show images of the evaluation dataset")
plot_images(train_dataset)

print("Show images of the evaluation dataset")
plot_images(test_dataset)

import tensorflow_datasets as tfds
import tensorflow as tf

def preprocess(features):
    image = tf.image.resize(features['input'], [128, 128])
    image = image / 255.0  
    return image, features['output']  

def create_dataloader(split='train[:80%]'):
    dataset, info = tfds.load('bee_dataset', as_supervised=False, with_info=True, split=split)


    dataset = dataset.map(preprocess)  
    dataset = dataset.shuffle(1000) if 'train' in split else dataset  
    dataset = dataset.batch(32)        
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  

    return dataset


train_dataloader = create_dataloader('train[:80%]')
test_dataloader = create_dataloader('train[80%:]')


for images, labels in train_dataloader.take(1):
    print("Training batch:")
    print("Image shape:", images.shape)  
    print("Labels:", labels)  


for images, labels in test_dataloader.take(1):
    print("Testing batch:")
    print("Image shape:", images.shape)  
    print("Labels:", labels) 

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# def build_model(num_classes):
#     model = Sequential([
#         # لایه کانولوشنی اول
#         Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#         MaxPooling2D((2, 2)),
#         # لایه کانولوشنی دوم
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         # لایه کانولوشنی سوم
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D((2, 2)),
#         # تبدیل نقشه‌های ویژگی به وکتور
#         Flatten(),
#         # لایه متراکم
#         Dense(128, activation='relu'),
#         Dropout(0.5),  # جلوگیری از بیش‌برازش
#         # لایه خروجی
#         Dense(num_classes, activation='softmax')
#     ])

#     # کامپایل مدل
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# # تعداد کلاس‌ها را بر اساس دیتاست تنظیم کنید
# num_classes = 3  # فرض بر این است که سه دسته مختلف زنبور وجود دارد
# model = build_model(num_classes)
# model.summary()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam

def build_multi_output_model(num_classes):
    input_layer = Input(shape=(128, 128, 3))

    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    cooling_output = Dense(1, activation='sigmoid', name='cooling_output')(x)
    pollen_output = Dense(1, activation='sigmoid', name='pollen_output')(x)
    varroa_output = Dense(1, activation='sigmoid', name='varroa_output')(x)
    wasps_output = Dense(1, activation='sigmoid', name='wasps_output')(x)

    model = Model(inputs=input_layer, outputs=[cooling_output, pollen_output, varroa_output, wasps_output])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'cooling_output': 'binary_crossentropy', 'pollen_output': 'binary_crossentropy',
                        'varroa_output': 'binary_crossentropy', 'wasps_output': 'binary_crossentropy'},
                  metrics={'cooling_output': 'accuracy', 'pollen_output': 'accuracy',
                           'varroa_output': 'accuracy', 'wasps_output': 'accuracy'})
    return model

model = build_multi_output_model(num_classes=1)
model.summary()


history = model.fit(train_dataloader, epochs=10)


results = model.evaluate(test_dataloader)
print("test loss, test acc:", results)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_dataset):
    # Defining the order of outputs as they appear in the model's predictions
    output_keys = ['cooling_output', 'pollen_output', 'varroa_output', 'wasps_output']
    y_true_dict = {key: [] for key in output_keys}
    y_pred_dict = {key: [] for key in output_keys}

    for images, labels in test_dataset:
        preds = model.predict(images)

        # Collecting predictions and true labels for each output
        for i, key in enumerate(output_keys):
            y_true_dict[key].extend(labels[key].numpy())
            y_pred_dict[key].extend((preds[i].squeeze() > 0.5).astype(int))

    # Calculating and printing the confusion matrix and classification report for each output
    for key in output_keys:
        print(f"Results for {key}:")
        y_true = y_true_dict[key]
        y_pred = y_pred_dict[key]
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {key}')
        plt.show()

# Now call evaluate_model with your model and test data loader
evaluate_model(model, test_dataloader)

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from collections import Counter

# Load the dataset and split it into training and testing sets
(train_dataset, test_dataset), ds_info = tfds.load(
    'bee_dataset',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=False,
    with_info=True
)

# Function to convert labels to a string
def label_to_string(label):
    label_str = []
    for key, value in label.items():
        if value.numpy() == 1.0:
            label_str.append(key.replace('_output', ''))
    return ', '.join(label_str) if label_str else 'No activity'

# Function to plot images
def plot_images(dataset, num_images=5):
    plt.figure(figsize=(15, 15))
    for i, example in enumerate(dataset.take(num_images)):
        ax = plt.subplot(3, 3, i + 1)
        image = example['input']  # Access the image
        label = example['output']  # Access the label
        label_text = label_to_string(label)  # Convert label to text
        plt.imshow(image)
        plt.title(f"Activity: {label_text}")
        plt.axis("off")
    plt.show()

# Function to compute and visualize the distribution of label combinations
def plot_label_distribution(dataset):
    label_counts = Counter()

    for example in dataset:
        label = label_to_string(example['output'])
        label_counts[label] += 1

    # Plotting the distribution of label combinations
    labels, counts = zip(*label_counts.items())
    plt.figure(figsize=(10, 8))
    plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Label Combinations')
    plt.title('Distribution of Label Combinations in Dataset')
    plt.show()

# Display images from the training dataset
print("Displaying images from the training dataset:")
plot_images(train_dataset)

# Display images from the testing dataset
print("Displaying images from the testing dataset:")
plot_images(test_dataset)

# Visualize the label distribution in the training dataset
print("Visualizing label distribution in the training dataset:")
plot_label_distribution(train_dataset)

import tensorflow_datasets as tfds

# Load the dataset and split it into training and testing sets
(train_dataset, test_dataset), ds_info = tfds.load(
    'bee_dataset',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=False,
    with_info=True
)

# Function to count the number of images in a dataset split
def count_images(dataset):
    return sum(1 for _ in dataset)

# Count images in the training and testing datasets
num_train_images = count_images(train_dataset)
num_test_images = count_images(test_dataset)
total_images = num_train_images + num_test_images

print(f"Number of images in the training dataset: {num_train_images}")
print(f"Number of images in the testing dataset: {num_test_images}")
print(f"Total number of images in the dataset: {total_images}")

