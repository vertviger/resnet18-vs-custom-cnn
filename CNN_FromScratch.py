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
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
    # forward method
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

# time measures

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
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform= transform_valid_img)
    
    train_loader = DataLoader(train_dataset, 
                                  batch_size = 64, 
                                  shuffle = True, 
                                  num_workers=2, 
                                  pin_memory=True)

    valid_loader = DataLoader(valid_dataset, 
                                  batch_size=64, 
                                  shuffle = False, 
                                  num_workers=2, 
                                  pin_memory=True)

    test_loader = DataLoader(test_dataset, 
                                      batch_size=64, 
                                      shuffle = False, 
                                      num_workers=2, 
                                      pin_memory=True)

    class_count = len(train_dataset.classes)
    print(f"Number of classes {class_count}: {train_dataset.classes}")

    weigths_path = 'best_cnn_weights.pth'

    model = CNNFromScratch(class_count).to(hardware)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr = 0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=16, factor=0.5)
    # let the user choose whether to retrain the model or just run test case
    choice = input("Select Y for training, N for test (Y/N): ").strip().upper()
    # Training and Test selected
    if choice == 'Y':
        start_time = time.perf_counter()
        
        # save best loss values for best weights update 
        best_val_loss = float('inf')
        if os.path.exists(weigths_path):
            checkpoint = torch.load(weigths_path, map_location=hardware)
            if isinstance(checkpoint, dict) and 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print("best validation loss was found from old runs")

        print("Starting training/validation phase")
        # training setup
        count = 0
        number_epochs = 200
        for epoch in range(number_epochs):
            count = count + 1
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
            total = 0
            correct = 0
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(hardware), labels.to(hardware)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                avg_val_loss = val_loss/len(valid_loader)
                scheduler.step(avg_val_loss)
                accuracy = 100*correct/total
                print(f"Epoch N{count} Validation accuracy: {accuracy}, Validation loss: {avg_val_loss}, Best Saved Loss: {best_val_loss}")
            # saving weights if the loss is better than saved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, weigths_path)
                print("Saved weights were updated due to better loss value")

        # time taken showcase
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Training took {elapsed_time:.2f} seconds")
    
    # Test only selected
    elif choice == 'N':
        print("Starting test phase")
        if os.path.exists(weigths_path):
            checkpoint = torch.load(weigths_path, map_location=hardware)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded saved weights")
        else:
            raise FileNotFoundError("No weights found, you must train model first!")
    else: 
        raise ValueError("Invalid selection!")

    # test phase
    model.eval()
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(hardware), labels.to(hardware)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
    
    accuracy = 100*correct/total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = train_dataset.classes)
    disp.plot(cmap = plt.cm.Greens)
    plt.title(f"Test confusion matrix, accuracy: {accuracy:.2f}%")
    plt.show()