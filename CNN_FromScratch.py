from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
import torch.nn as nn 
from torch import optim
import time

# CNN class
class CNNFromScratch(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNNFromScratch, self).__init__()
        self.features = nn.Sequential(
            # Layer 1 3 -> 32 | 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            # Layer 2 32 -> 64 | 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 3 64 -> 128 | 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Layer 4 128 -> 256 | 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Average pooling 14 -> 1
            nn.AdaptiveAvgPool2d((1,1))
            # Output dimensions 256 x 1 x 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    # forward method
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

# time measures
    start_time = time.perf_counter()

    # used hardware check
    hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training using: {hardware}")
    # images transformations Imagenet means and stds
    transform_tran_img = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    transform_valid_img = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    dataset_dir = "./dataset"
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, "train"), transform = transform_tran_img)
    valid_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, "valid"), transform= transform_valid_img)

    train_loader = DataLoader(train_dataset, 
                                  batch_size = 32, 
                                  shuffle = True, 
                                  num_workers=2, 
                                  pin_memory=True)

    valid_loader = DataLoader(valid_dataset, 
                                  batch_size=32, 
                                  shuffle = False, 
                                  num_workers=2, 
                                  pin_memory=True)

    class_count = len(train_dataset.classes)
    print(f"Number of classes {class_count}: {train_dataset.classes}")

    model = CNNFromScratch(class_count)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 0.001, weight_decay=1e-4)
    # training setup
    number_epochs = 30
    for epoch in range(number_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(hardware), labels.to(hardware)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
    # validation loop
    model.eval()
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0

    val_losses = []
    val_accuracies = []
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(hardware), labels.to(hardware)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
        accuracy = 100*correct/total
        print(f"Validation accuracy: {accuracy}")

    # time taken showcase
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Process took {elapsed_time:.2f} seconds")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = train_dataset.classes)
    disp.plot(cmap = plt.cm.Greens)
    plt.show()