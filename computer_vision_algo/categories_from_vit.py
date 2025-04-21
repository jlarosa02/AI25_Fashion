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

styles_df = styles_df[styles_df['year'] == 2012] # clean up the data: only year 2012
print(styles_df.shape)

styles_df = styles_df[styles_df['masterCategory'] == "Apparel"] # clean up the data: only keep apparel

# make sure every row has a matching image: add a column with the image path
styles_df['imagePath'] = styles_df['id'].apply(lambda x: str(x) + '.jpg')
styles_df['imagePath'] = styles_df['imagePath'].apply(lambda x: './data/images/' + x)
styles_df = styles_df[styles_df['imagePath'].apply(os.path.exists)]
print(styles_df.shape) # only rows with valid image paths

# exploring data classes
gender_data = styles_df['gender'].unique()
article_data = styles_df['articleType'].unique()
color_data = styles_df['baseColour'].unique()
season_data = styles_df['season'].unique()
usage_data = styles_df['usage'].unique()
# print(gender_data, article_data, color_data, season_data, usage_data)

# encoding each category into numerical values models
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

# exact df that will be given to the model
model_df = pd.DataFrame({
    'id': styles_df['id'],
    'imagePath': styles_df['imagePath'],
    'gender': gender_encoded,
    'articleType': article_encoded,
    'baseColour': color_encoded,
    'season': season_encoded,
    'usage': usage_encoded
})

print(model_df.head())

# preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize the image to 224x224 for ViT model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imageNet standards
])

batch_size = 64

# create training and test datasets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(model_df, test_size=0.2, random_state=42)

trainset = FashionProductDataset(df=train_df, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = FashionProductDataset(df=test_df, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# for each category, the number of unique values in the encoder is the number of classes
model = ViT_Model(
    num_genders= len(gender_encoder.classes_),
    num_articleTypes= len(article_encoder.classes_),
    num_baseColours= len(color_encoder.classes_),
    num_seasons= len(season_encoder.classes_),
    num_usages= len(usage_encoder.classes_)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

total_steps = len(trainloader) * 20 # 20 epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"): # tqdm shows porgress bar
        inputs = batch['image'].to(device)
        gender_labels = batch['gender'].to(device)
        article_labels = batch['articleType'].to(device)
        color_labels = batch['baseColour'].to(device)
        season_labels = batch['season'].to(device)
        usage_labels = batch['usage'].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # compare the loss of each category to the labels per each category
        gender_loss = criterion(outputs['gender'], gender_labels)
        article_loss = criterion(outputs['article_type'], article_labels)
        color_loss = criterion(outputs['color'], color_labels)
        season_loss = criterion(outputs['season'], season_labels)
        usage_loss = criterion(outputs['usage'], usage_labels)

        loss = gender_loss + article_loss + color_loss + season_loss + usage_loss
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    loss_per_epoch = running_loss / total_samples

    return loss_per_epoch
    
def test_epoch(model, dataloader, criterion, device) :
    model.eval()
    running_loss = 0
    gender_correct = 0
    article_correct = 0
    color_correct = 0
    season_correct = 0
    usage_correct = 0
    total_samples = 0

    with torch.no_grad():
            for batch in tqdm(dataloader, desc='Testing'):
                inputs = batch['image'].to(device)
                gender_labels = batch['gender'].to(device)
                article_labels = batch['articleType'].to(device)
                color_labels = batch['baseColour'].to(device)
                season_labels = batch['season'].to(device)
                usage_labels = batch['usage'].to(device)
                
                outputs = model(inputs)

                # compare the loss of each category to the labels per each category
                gender_loss = criterion(outputs['gender'], gender_labels)
                article_loss = criterion(outputs['article_type'], article_labels)
                color_loss = criterion(outputs['color'], color_labels)
                season_loss = criterion(outputs['season'], season_labels)
                usage_loss = criterion(outputs['usage'], usage_labels)

                loss = gender_loss + article_loss + color_loss + season_loss + usage_loss
                
                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

                _, gender_predicted = torch.max(outputs['gender'], 1)
                _, article_predicted = torch.max(outputs['article_type'], 1)
                _, color_predicted = torch.max(outputs['color'], 1)
                _, season_predicted = torch.max(outputs['season'], 1)
                _, usage_predicted = torch.max(outputs['usage'], 1)

                # verifying if prediction is correct against label
                gender_correct += (gender_predicted == gender_labels).sum().item()
                article_correct += (article_predicted == article_labels).sum().item()
                color_correct += (color_predicted == color_labels).sum().item()
                season_correct += (season_predicted == season_labels).sum().item()
                usage_correct += (usage_predicted == usage_labels).sum().item()

    avg_loss = running_loss / total_samples
    gender_accuracy = gender_correct / total_samples
    article_accuracy = article_correct / total_samples
    color_accuracy = color_correct / total_samples
    season_accuracy = season_correct / total_samples
    usage_accuracy = usage_correct / total_samples

    return avg_loss, gender_accuracy, article_accuracy, color_accuracy, season_accuracy, usage_accuracy

def train_model(model, train_loader, test_loader, critertion, optimizer, scheduler, best_model_path, num_epochs=3):
    train_losses = []
    test_losses = []
    accuracies = {'gender': [], 'article': [], 'color': [], 'season': [], 'usage': []}

    best_val_loss = float('inf')

    # overwrite with the new metrics
    with open(f'{best_model_path}_metrics.json', 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': accuracies
        }, f)

    for epoch in range(num_epochs):
        train_epoch_loss = train_epoch(model, train_loader, critertion, optimizer, device)
        test_epoch_loss, test_gender_accuracy, test_article_accuracy, test_color_accuracy, test_season_accuracy, test_usage_accuracy = test_epoch(model, test_loader, criterion, device)

        scheduler.step(test_epoch_loss)

        # total losses across all epochs
        train_losses.append(train_epoch_loss) 
        test_losses.append(test_epoch_loss)
        accuracies['gender'].append(test_gender_accuracy)
        accuracies['article'].append(test_article_accuracy)
        accuracies['color'].append(test_color_accuracy)
        accuracies['season'].append(test_season_accuracy)
        accuracies['usage'].append(test_usage_accuracy)

        # Save the updated metrics to the JSON file
        with open(f'{best_model_path}_metrics.json', 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': test_losses,
                'val_accuracies': accuracies
            }, f)
        print(f"Writing metrics to JSON: {train_losses}, {test_losses}, {accuracies}")

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {train_epoch_loss:.3f}')
        print(f'Test Loss: {test_epoch_loss:.3f}')
        print(f'Test Accuracy - gender: {test_gender_accuracy:.4f}, article type: {test_article_accuracy:.4f}, color: {test_color_accuracy:.4f}, season: {test_season_accuracy:.4f}, usage: {test_usage_accuracy:.4f}')

        if test_epoch_loss < best_val_loss:
            
            # save new best loss
            best_val_loss = test_epoch_loss

            # save the model
            torch.save(model.state_dict(), f'{best_model_path}.pth')
            print('newly saved model')

    return train_losses, test_losses, accuracies

# train_loss_overall, test_loss_overall, accuracy_overall = train_model(model, trainloader, testloader, criterion, optimizer, scheduler, best_model_path='best_model', num_epochs=20)
