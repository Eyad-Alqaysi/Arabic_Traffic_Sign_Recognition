import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score

# Define your data transform
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomRotation(20), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Load your data
train_dataset = datasets.ImageFolder(root='../../main_dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='../../main_dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained MobileNet model
model = timm.create_model('mobilevitv2_075.cvnets_in1k', pretrained=True)
num_classes = len(train_dataset.classes)

# Replace the classifier layer
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Check if CUDA is available and get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Define the best accuracy
best_accuracy = 0
epsilon = 1e-5
epochs = 10

# Training loop
print('Start training...')
for epoch in range(epochs): 
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    current_accuracy = (100 * correct / total)
    print(f'\tEpoch {epoch+1}, Accuracy: {100 * current_accuracy}%, Loss: {loss.item()}, F1-score: {f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average="macro")}')

    if current_accuracy + epsilon >= best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), 'models/best.pt')

# Save the last model
torch.save(model.state_dict(), 'models/last.pt')