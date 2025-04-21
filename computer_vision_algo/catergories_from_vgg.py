import json
import os
from sklearn.calibration import LabelEncoder
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

styles_df = pd.read_csv('./data/styles.csv', on_bad_lines='skip')
print(styles_df.shape)

styles_df.dropna(inplace = True)  # clean up the data: remove rows with NaN values
print(styles_df.shape)

styles_df = styles_df[styles_df['masterCategory'] == "Apparel"] # clean up the data: only keep apparel

# make sure every row has a matching image: add a column with the image path
styles_df['imagePath'] = styles_df['id'].apply(lambda x: str(x) + '.jpg')
styles_df['imagePath'] = styles_df['imagePath'].apply(lambda x: '/kaggle/input/fashion-product-images-small/images/' + x)
styles_df = styles_df[styles_df['imagePath'].apply(os.path.exists)]
print(styles_df.shape) # only rows with valid image paths

gender_distribution = styles_df['gender'].value_counts().to_dict()
print(gender_distribution)

gender_sampling_strategy = {
    'Women': 5000,
    'Men': 5000
}

# undersampling needed for VGG-16 version to prevent overfitting 
for gender, count in gender_distribution.items():
    if gender not in gender_sampling_strategy:
        gender_sampling_strategy[gender] = count

undersampler_gender = RandomUnderSampler(sampling_strategy=gender_sampling_strategy, random_state=42)

x_gender, y_gender = undersampler_gender.fit_resample(styles_df[['id']], styles_df['gender'])
undersampled_gender_df = pd.DataFrame({'id': x_gender['id'].to_numpy().flatten(), 'gender': y_gender})

print(undersampled_gender_df['gender'].value_counts())

articleType_distribution = styles_df['articleType'].value_counts().to_dict()
print(articleType_distribution)

articleType_sampling_strategy = {
    'Tshirts': 1000,
    'Shirts': 1000,
}

for articleType, count in articleType_distribution.items():
    if articleType not in articleType_sampling_strategy:
        articleType_sampling_strategy[articleType] = count

undersampler_article = RandomUnderSampler(sampling_strategy=articleType_sampling_strategy, random_state=42)

x_articleType, y_articleType = undersampler_article.fit_resample(styles_df[['id']], styles_df['articleType'])
undersampled_articleType_df = pd.DataFrame({'id': x_articleType['id'].to_numpy().flatten(), 'articleType': y_articleType})

print(undersampled_articleType_df['articleType'].value_counts())

color_distribution = styles_df['baseColour'].value_counts().to_dict()
print(color_distribution)

color_sampling_strategy = {
    'Black': 2000,
    'White': 2000,
    'Blue': 2000,
}

for color, count in color_distribution.items():
    if color not in color_sampling_strategy:
        color_sampling_strategy[color] = count

undersampler_color = RandomUnderSampler(sampling_strategy=color_sampling_strategy, random_state=42)

x_color, y_color = undersampler_color.fit_resample(styles_df[['id']], styles_df['baseColour'])
undersampled_color_df = pd.DataFrame({'id': x_color['id'].to_numpy().flatten(), 'baseColour': y_color})

print(undersampled_color_df['baseColour'].value_counts())

season_distribution = styles_df['season'].value_counts().to_dict()
print(season_distribution)

season_sampling_strategy = {
    'Summer': 5000,
    'Fall': 5000,
}

for season, count in season_distribution.items():
    if season not in season_sampling_strategy:
        season_sampling_strategy[season] = count

undersampler_season = RandomUnderSampler(sampling_strategy=season_sampling_strategy, random_state=42)

x_season, y_season = undersampler_season.fit_resample(styles_df[['id']], styles_df['season'])
undersampled_season_df = pd.DataFrame({'id': x_season['id'].to_numpy().flatten(), 'season': y_season})

print(undersampled_season_df['season'].value_counts())

usage_distribution = styles_df['usage'].value_counts().to_dict()
print(usage_distribution)

usage_sampling_strategy = {
    'Casual': 7000,
}

for usage, count in usage_distribution.items():
    if usage not in usage_sampling_strategy:
        usage_sampling_strategy[usage] = count

undersampler_usage = RandomUnderSampler(sampling_strategy=usage_sampling_strategy, random_state=42)

x_usage, y_usage = undersampler_usage.fit_resample(styles_df[['id']], styles_df['usage'])
undersampled_usage_df = pd.DataFrame({'id': x_usage['id'].to_numpy().flatten(), 'usage': y_usage})

print(undersampled_usage_df['usage'].value_counts())

merged_df = undersampled_gender_df.merge(undersampled_articleType_df, on='id', how='inner')
merged_df = merged_df.merge(undersampled_color_df, on='id', how='inner')
merged_df = merged_df.merge(undersampled_season_df, on='id', how='inner')
merged_df = merged_df.merge(undersampled_usage_df, on='id', how='inner')

undersampled_df = pd.merge(merged_df, styles_df[['id', 'imagePath']], on='id', how='inner')

print(undersampled_df.shape)

# encoding each category for model
gender_encoder = LabelEncoder()
article_encoder = LabelEncoder()
color_encoder = LabelEncoder()
season_encoder = LabelEncoder()
usage_encoder = LabelEncoder()

gender_encoded = gender_encoder.fit_transform(undersampled_df['gender'])
article_encoded = article_encoder.fit_transform(undersampled_df['articleType'])
color_encoded = color_encoder.fit_transform(undersampled_df['baseColour'])
season_encoded = season_encoder.fit_transform(undersampled_df['season'])
usage_encoded = usage_encoder.fit_transform(undersampled_df['usage'])

# exact df that will be given to the model
model_df = pd.DataFrame({
    'id': undersampled_df['id'],
    'imagePath': undersampled_df['imagePath'],
    'gender': gender_encoded,
    'articleType': article_encoded,
    'baseColour': color_encoded,
    'season': season_encoded,
    'usage': usage_encoded
})

print(model_df.head())

output_signature = (
       tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # For images
       {
           'gender_output': tf.TensorSpec(shape=(None, len(gender_encoder.classes_)), dtype=tf.float32),
           'articleType_output': tf.TensorSpec(shape=(None, len(article_encoder.classes_)), dtype=tf.float32),
           'color_output': tf.TensorSpec(shape=(None, len(color_encoder.classes_)), dtype=tf.float32),
           'season_output': tf.TensorSpec(shape=(None, len(season_encoder.classes_)), dtype=tf.float32),
           'usage_output': tf.TensorSpec(shape=(None, len(usage_encoder.classes_)), dtype=tf.float32)
       }
   )

def multi_output_generator(image_generator):
    for images, label_batch in image_generator:
        # Get the corresponding labels for the current batch
        labels_dict = {
            'gender_output': tf.keras.utils.to_categorical(label_batch[0], num_classes=len(gender_encoder.classes_)),
            'articleType_output': tf.keras.utils.to_categorical(label_batch[1], num_classes=len(article_encoder.classes_)),
            'color_output': tf.keras.utils.to_categorical(label_batch[2], num_classes=len(color_encoder.classes_)),
            'season_output': tf.keras.utils.to_categorical(label_batch[3], num_classes=len(season_encoder.classes_)),
            'usage_output': tf.keras.utils.to_categorical(label_batch[4], num_classes=len(usage_encoder.classes_)),
        }

        yield images, labels_dict

# preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224 for ViT model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imageNet standards
])

batch_size = 16

# create training and test datasets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)

trainset = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
trainloader = trainset.flow_from_dataframe(
    dataframe=train_df,
    x_col='imagePath',
    y_col=['gender', 'articleType', 'baseColour', 'season', 'usage'],
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='multi_output',
)

testset = ImageDataGenerator()
testloader = testset.flow_from_dataframe(
    dataframe=test_df,
    x_col='imagePath',
    y_col=['gender', 'articleType', 'baseColour', 'season', 'usage'],
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='multi_output',
)

# Input layer
inputs = tf.keras.Input(shape=(224, 224, 3))
print("Type of inputs:", type(inputs))

print("Type of layers:", type(layers))

# VGG-like convolutional base
print("Type of layers.Conv2D:", type(layers.Conv2D))
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
print("Type after first Conv2D:", type(x))
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

# Flatten and fully connected layers
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)

# Output layers for each category
gender_output = layers.Dense(len(gender_encoder.classes_), activation='softmax', name='gender_output')(x)
articleType_output = layers.Dense(len(article_encoder.classes_), activation='softmax', name='articleType_output')(x)
color_output = layers.Dense(len(color_encoder.classes_), activation='softmax', name='color_output')(x)
season_output = layers.Dense(len(season_encoder.classes_), activation='softmax', name='season_output')(x)
usage_output = layers.Dense(len(usage_encoder.classes_), activation='softmax', name='usage_output')(x)

# Define the model
vggmodel = Model(
    inputs=inputs,
    outputs={
        'gender_output': gender_output,
        'articleType_output': articleType_output,
        'color_output': color_output,
        'season_output': season_output,
        'usage_output': usage_output
    }
)

optimizer = Adam(learning_rate=0.0001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Compile the model
vggmodel.compile(optimizer=optimizer,
                 loss={'gender_output': 'categorical_crossentropy',
                       'articleType_output': 'categorical_crossentropy',
                       'color_output': 'categorical_crossentropy',
                       'season_output': 'categorical_crossentropy',
                       'usage_output': 'categorical_crossentropy'},
                    metrics={
                     'gender_output': ['accuracy'],
                     'articleType_output': ['accuracy'],
                     'color_output': ['accuracy'],
                     'season_output': ['accuracy'],
                     'usage_output': ['accuracy']
                 },
                 )

checkpoint = ModelCheckpoint('best_model_vgg.h5', monitor='val_color_output_accuracy', save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', patience=5, mode='min')

train_dataset = tf.data.Dataset.from_generator(
    lambda: multi_output_generator(trainloader),
    output_signature=output_signature
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: multi_output_generator(testloader),
    output_signature=output_signature
)

history = vggmodel.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    steps_per_epoch=len(train_df) // batch_size,
    validation_steps=len(test_df) // batch_size,
    callbacks=[checkpoint, early, lr_scheduler]
)

# Extract metrics
metrics = {
    "total_training_loss": history.history.get('loss', []),  # Total training loss per epoch
    "total_validation_loss": history.history.get('val_loss', []),  # Total validation loss per epoch
    "validation_accuracy_per_category": {
        "gender_output": history.history.get('val_gender_output_accuracy', []),
        "articleType_output": history.history.get('val_articleType_output_accuracy', []),
        "color_output": history.history.get('val_color_output_accuracy', []),
        "season_output": history.history.get('val_season_output_accuracy', []),
        "usage_output": history.history.get('val_usage_output_accuracy', [])
    }
}

# Save metrics to a JSON file
with open('vgg_metrics_final.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to vgg_metrics_final.json")
