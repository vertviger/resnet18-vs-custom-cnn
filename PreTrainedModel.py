import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn 
from torch import optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

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
                              num_workers=8, 
                              pin_memory=True)

    valid_loader = DataLoader(valid_dataset, 
                              batch_size=32, 
                              shuffle = False, 
                              num_workers=8, 
                              pin_memory=True)

    class_count = len(train_dataset.classes)
    print(f"Number of classes {class_count}: {train_dataset.classes}")
    # pretrained model setup
    pretrained_model = resnet18(weights = ResNet18_Weights.DEFAULT)
    for param in pretrained_model.parameters():
        param.requires_grad = False
    # unfreezing the backbone
    for param in pretrained_model.layer4.parameters():
        param.requires_grad = True
    for param in pretrained_model.fc.parameters():
        param.requires_grad = True
    # replacing the head
    in_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(in_features, class_count)
    pretrained_model = pretrained_model.to(hardware)
    # optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': pretrained_model.layer4.parameters(), 'lr': 1e-4},
        {'params': pretrained_model.fc.parameters(), 'lr': 1e-3}
    ], lr=1e-4)
    # pretrained model training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        pretrained_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(hardware), labels.to(hardware)
            optimizer.zero_grad()
            outputs = pretrained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # pretrained model validation loop
    pretrained_model.eval()
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(hardware), labels.to(hardware)
            outputs = pretrained_model(images)
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

    # confusion matrix on validation set
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
    disp.plot(cmap=plt.cm.Greens)
    plt.show()