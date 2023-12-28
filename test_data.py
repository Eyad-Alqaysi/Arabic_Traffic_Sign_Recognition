import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, f1_score
import timm

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your model
model = timm.create_model('mobilevitv2_075.cvnets_in1k', pretrained=True)
state_dict = torch.load('./99.87/best.pt')
try:
    model.load_state_dict(state_dict)
except RuntimeError as e:
    print("Strict loading failed, attempting partial loading.")
    model.load_state_dict(state_dict, strict=False)

model.to(device)
model.eval()

# Prepare your test dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    # Remove augmentation transformations like RandomRotation and ColorJitter for testing
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder(root='../main_dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print('Start testing...\n')

# Run the model on the test data and calculate accuracy and F1 score
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')  # or 'macro'
f1_macro = f1_score(all_labels, all_preds, average='macro')

print(f"Accuracy: {accuracy}")
print(f"F1 Weighted Score: {f1}")
print(f"F1 Macro Score: {f1_macro}")
print(f"Correct: {sum(1 for i, j in zip(all_preds, all_labels) if i == j)}")
print(f"Total: {len(all_preds)}")