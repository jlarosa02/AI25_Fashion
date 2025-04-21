import json
import matplotlib.pyplot as plt

# Load the metrics from the file
with open('best_model_metrics.json', 'r') as f:
    metrics = json.load(f)
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    val_accuracies = metrics['val_accuracies']

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time For Vision Transformer Model')
plt.xticks(range(0, len(train_losses), 2))
plt.grid(True)
plt.legend()
plt.show()

categories = ['gender', 'article type', 'color', 'season', 'usage']
# Define a list of markers to use for each category
markers = ['o', 's', 'D', '^', 'x']  # Circle, square, diamond, triangle, cross

# Plot all accuracies for each category on one graph
plt.figure(figsize=(12, 6))
for i, category in enumerate(categories):
    plt.plot(val_accuracies[category], label=f'Accuracy ({category})', marker=markers[i % len(markers)])

# Add labels, title, grid, and legend
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy Over Time For Vision Transformer Model')
plt.xticks(range(0, len(train_losses), 2))
plt.grid(True)
plt.legend(loc='best')
plt.show()
    
# Load the metrics from the file
with open('vgg_metrics_final.json', 'r') as f:
    metrics = json.load(f)
    train_losses = metrics['total_training_loss']
    val_losses = metrics['total_validation_loss']
    val_accuracies = metrics['validation_accuracy_per_category']

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time For VGG16 Model')
plt.xticks(range(0, len(train_losses), 5))
plt.grid(True)
plt.legend()
plt.show()

categories = ['gender', 'article type', 'color', 'season', 'usage']
# Define a list of markers to use for each category
markers = ['o', 's', 'D', '^', 'x']  # Circle, square, diamond, triangle, cross

# Plot all accuracies for each category on one graph
plt.figure(figsize=(12, 6))
for i, category in enumerate(categories):
    plt.plot(val_accuracies[category], label=f'Accuracy ({category})', marker=markers[i % len(markers)])

# Add labels, title, grid, and legend
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy Over Time For VGG16 Model')
plt.xticks(range(0, len(train_losses), 5))
plt.grid(True)
plt.legend(loc='best')
plt.show()
