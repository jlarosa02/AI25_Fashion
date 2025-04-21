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
from ViT.vit import ViT
from fashion_dataset import *
from fashion_vit import ViT_Model
import tensorflow as tf
import cv2
import numpy as np
import keras,os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from transformers import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

styles_df = pd.read_csv('./data/styles.csv', on_bad_lines='skip')
print(styles_df.shape)

styles_df.dropna(inplace = True)  # clean up the data: remove rows with NaN values
print(styles_df.shape)

styles_df = styles_df[styles_df['year'] == 2012] # clean up the data: remove rows with year < 2016
print(styles_df.shape)

styles_df = styles_df[styles_df['masterCategory'] == "Apparel"] # clean up the data: only keep apparel

# make sure every row has a matching image: add a column with the image path
styles_df['imagePath'] = styles_df['id'].apply(lambda x: str(x) + '.jpg')
styles_df['imagePath'] = styles_df['imagePath'].apply(lambda x: './data/images/' + x)
styles_df = styles_df[styles_df['imagePath'].apply(os.path.exists)]
print(styles_df.shape) # only rows with valid image paths

gender_encoder = LabelEncoder()
article_encoder = LabelEncoder()
color_encoder = LabelEncoder()
season_encoder = LabelEncoder()
usage_encoder = LabelEncoder()

gender_encoded = gender_encoder.fit_transform(styles_df['gender'])
article_encoded = article_encoder.fit_transform(styles_df['articleType'])
color_encoded = color_encoder.fit_transform(styles_df['baseColour'])
season_encoded = season_encoder.fit_transform(styles_df['season'])
usage_encoded = usage_encoder.fit_transform(styles_df['usage'])

# df same as the one model was trained on; will extract random samples from it 
model_df = pd.DataFrame({
    'id': styles_df['id'],
    'imagePath': styles_df['imagePath'],
    'gender': gender_encoded,
    'articleType': article_encoded,
    'baseColour': color_encoded,
    'season': season_encoded,
    'usage': usage_encoded
})

# pre-process image inputs
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224 for ViT model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imageNet standards
])

# what the model architecture looks like 
vitmodel =  ViT_Model(
    num_genders= len(gender_encoder.classes_),
    num_articleTypes= len(article_encoder.classes_),
    num_baseColours= len(color_encoder.classes_),
    num_seasons= len(season_encoder.classes_),
    num_usages= len(usage_encoder.classes_)
)

checkpoint_path = "best_model.pth"
vitmodel.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

def prediction(model, path, transform, device):
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(image)

        _, predicted_gender = torch.max(outputs['gender'], 1)
        _, predicted_article = torch.max(outputs['article_type'], 1)
        _, predicted_color = torch.max(outputs['color'], 1)
        _, predicted_season = torch.max(outputs['season'], 1)
        _, predicted_usage = torch.max(outputs['usage'], 1)
    
    gender = gender_encoder.inverse_transform([predicted_gender.item()])[0]
    article = article_encoder.inverse_transform([predicted_article.item()])[0]
    color = color_encoder.inverse_transform([predicted_color.item()])[0]
    season = season_encoder.inverse_transform([predicted_season.item()])[0]
    usage = usage_encoder.inverse_transform([predicted_usage.item()])[0]

    return {
        'gender' : gender,
        'article type' : article,
        'color' : color,
        'season' : season,
        'usage' : usage
    }

predict_df = model_df.sample(n=4, random_state=42)

fig, axes = plt.subplots(4, 1, figsize=(18, 12))
fig.subplots_adjust(wspace=0.5)
axes = axes.flatten() # 2D array to 1D of 5 sample images

for i, ax in enumerate(axes):
    image_path = predict_df.iloc[i]['imagePath']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    ax.imshow(image)
    ax.axis('off')

    encoded_correct_labels = {
        'encoded gender' : predict_df.iloc[i]['gender'],
        'encoded articleType' : predict_df.iloc[i]['articleType'],
        'encoded color' : predict_df.iloc[i]['baseColour'],
        'encoded season' : predict_df.iloc[i]['season'],
        'encoded usage' : predict_df.iloc[i]['usage']
    }

    readable_correct_labels = {
        'gender' : gender_encoder.inverse_transform([encoded_correct_labels['encoded gender']])[0],
        'article type' : article_encoder.inverse_transform([encoded_correct_labels['encoded articleType']])[0],
        'color' : color_encoder.inverse_transform([encoded_correct_labels['encoded color']])[0],
        'season' : season_encoder.inverse_transform([encoded_correct_labels['encoded season']])[0],
        'usage' : usage_encoder.inverse_transform([encoded_correct_labels['encoded usage']])[0]
    }

    readable_predicted_labels = prediction(vitmodel, image_path, transform, device)

    correct_labels_caption = 'Correct: ' + ', '.join([f"{key}: {value}" for key, value in readable_correct_labels.items()])
    predicted_labels_caption = 'Predicted: ' + ', '.join([f"{key}: {value}" for key, value in readable_predicted_labels.items()])

    ax.set_title(f"{correct_labels_caption}\n{predicted_labels_caption}", fontsize=10)

plt.tight_layout()
plt.show()