# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:32:19 2024

@author: b.mariet.0711
"""

import os
import re
import pandas as pd
import psutil


def BuildDataframe(directory):
    data = {'image_path': [], 'label': []}
    for image_file in os.listdir(directory):
        # Vérifier avec .lower() pour accepter JPG, jpg, JPEG, etc.
        if image_file.lower().endswith(('.jpg', '.jpeg')):  
            try:
                category = int(image_file.split('_')[0])
                label = category - 1
                image_path = os.path.join(directory, image_file)
                data['image_path'].append(image_path)
                data['label'].append(label)
            except ValueError as e:
                print(f"Erreur avec le fichier {image_file}: {e}")
                continue
                
    print(f"Nombre d'images extraites : {len(data['image_path'])}")
    return pd.DataFrame(data)


train_dir = 'D:/Plants_2/train'
test_dir = 'D:/Plants_2/test'
valid_dir = 'D:/Plants_2/valid'

def check_directory_content(directory):
    print(f"\nVérification du répertoire : {directory}")
    if not os.path.exists(directory):
        print(f"Le répertoire n'existe pas : {directory}")
        return
        
    files = os.listdir(directory)
    print(f"Nombre total de fichiers : {len(files)}")
    if files:
        print("Premiers 5 fichiers trouvés :")
        for file in files[:5]:
            print(f" - {file}")
    else:
        print("Aucun fichier trouvé dans le répertoire")

# Vérifier chaque répertoire
check_directory_content(train_dir)
check_directory_content(test_dir)
check_directory_content(valid_dir)

train_dataframe = BuildDataframe(train_dir)


test_dataframe = BuildDataframe(test_dir)


valid_dataframe = BuildDataframe(valid_dir)

sample_df = train_dataframe.sample(n=10)
print(sample_df)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU config OK")
else:
    print("Error using GPU, CPU will be used")
tf.keras.backend.clear_session()

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model



memory_info = psutil.virtual_memory()
print(f"RAM utilisée : {memory_info.used / (1024 ** 3):.2f} GB / {memory_info.total / (1024 ** 3):.2f} GB")

input_shape = (224, 224, 3)
num_classes = 22
batch_size = 32
epochs = 10


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

print(train_dataframe)

train_directory = 'D:/Plants_2/train'
test_directory = 'D:/Plants_2/test'
valid_directory = 'D:/Plants_2/valid'

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_dataframe,
    directory=train_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='raw'  # Pour les labels numériques
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory=test_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='raw'
)

valid_generator = test_datagen.flow_from_dataframe(
    dataframe=valid_dataframe,
    directory=valid_directory,
    x_col='image_path',
    y_col='label',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='raw'
)

base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

# Bloquer la partie convolutionnelle du modèle pré-entraîné
for layer in base_model.layers:
    layer.trainable = False

# Ajouter les nouvelles couches pour la classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Construction du modèle final
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

import time

# Mesure du temps d'entraînement
start_time = time.time()

history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator)

training_time = time.time() - start_time
print(f"Temps d'entraînement : {training_time:.2f} secondes")

# Sauvegarde du modèle
model.save('D:/model/modele_feuilles.h5')
print("Modèle sauvegardé avec succès.")


# Mesure du temps d'inférence
start_time = time.time()

test_loss, test_acc = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

inference_time = time.time() - start_time
print(f"Temps d'inférence : {inference_time:.2f} secondes par lot")

import matplotlib.pyplot as plt

# Courbes de précision
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Courbes de perte
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results_summary = {
    'Metric': ['Accuracy', 'Training Time (s)', 'Inference Time (s)', 'Memory Usage (GB)'],
    'Value': [test_acc, training_time, inference_time, memory_info.used / (1024 ** 3)]
}

df_results = pd.DataFrame(results_summary)
print(df_results)



