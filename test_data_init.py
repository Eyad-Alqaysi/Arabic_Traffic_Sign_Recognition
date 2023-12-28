import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score as f1_score_function  # renamed to avoid name clash
import os

print('Start testing...')
print(os.getcwd())

# Data transform
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomRotation(20), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Load data
test_dataset = datasets.ImageFolder(root='../main_dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load and prepare model
model = timm.create_model('mobilevitv2_075.cvnets_in1k', pretrained=True)
num_classes = len(test_dataset.classes)

# Load the model weights with strict=False, but ignore the final layer
state_dict = torch.load('./99.87/best.pt')
model.load_state_dict(state_dict, strict=False)

# Prepare for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print('Start testing...\n')
# Test the model
correct = 0
total = 0
f1_accumulator = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        f1_accumulator += f1_score_function(labels.cpu(), predicted.cpu(), average='macro')
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
f1_avg = f1_accumulator / len(test_loader)
print(f"Accuracy: {accuracy}%, Correct: {correct}, Total: {total}, F1 Score: {f1_avg}")
