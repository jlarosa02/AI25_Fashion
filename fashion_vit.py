import torch
import torchvision
import timm
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import ViTModel, ViTConfig

class ViT_Model(nn.Module):
    def __init__(self, num_genders, num_articleTypes, num_baseColours, num_seasons, num_usages):
        super().__init__()
        # vision transformer model
        # https://huggingface.co/docs/transformers/model_doc/vit
        
        # ViT processes input image and outputs sequence of embeddings per image patch (each embedding is a vector the size of hidden size)
        # hidden_size of 768 is default config
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # divide layers into 5 classifiers
        # linear claissifiers will take embedding of CLS token (first token in sequence) as input
        # the size of the CLS token is the same as hidden layer output size
        # linear classifiers will output logits for each class: vector of size num_classes
        self.gender_classifier = nn.Linear(self.vit.config.hidden_size, num_genders)
        self.article_classifier = nn.Linear(self.vit.config.hidden_size, num_articleTypes)
        self.color_classifier = nn.Linear(self.vit.config.hidden_size, num_baseColours)
        self.season_classifier = nn.Linear(self.vit.config.hidden_size, num_seasons)
        self.usage_classifier = nn.Linear(self.vit.config.hidden_size, num_usages)

    def forward(self, x):
        output = self.vit(x) # shape is (batch_size, sequence_length, hidden_size)
        # CLS token is first token in the sequence and is used for classification 

        last_hidden_state = output.last_hidden_state

        # Use the [CLS] token (first token) for classification
        cls_token_embedding = last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

        gender_output = self.gender_classifier(cls_token_embedding)
        article_output = self.article_classifier(cls_token_embedding)
        color_output = self.color_classifier(cls_token_embedding)
        season_output = self.season_classifier(cls_token_embedding)
        usage_output = self.usage_classifier(cls_token_embedding)

        return {
            'gender': gender_output,
            'article_type': article_output,
            'color': color_output,
            'season': season_output,
            'usage': usage_output
        }
    
