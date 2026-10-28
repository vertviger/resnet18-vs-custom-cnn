import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn 
from torch import optim
import time

from set_seed import set_seed
from test_and_visualize_model import test_and_visalize

if __name__ == '__main__':
    # seed setting 
    set_seed()
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
    test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, "test"), transform= transform_valid_img)

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
    
    test_loader = DataLoader(test_dataset, 
                                batch_size=64, 
                                shuffle = False, 
                                num_workers=2, 
                                pin_memory=True)

    class_count = len(train_dataset.classes)
    print(f"Number of classes {class_count}: {train_dataset.classes}")

    weigths_path = 'best_resnet18_F_weights.pth'

    # pretrained model setup
    pretrained_model = resnet18(weights = ResNet18_Weights.DEFAULT)
    for param in pretrained_model.parameters():
        param.requires_grad = False

    in_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(in_features, class_count)
    pretrained_model = pretrained_model.to(hardware)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(pretrained_model.fc.parameters(), lr = 0.001)

    # let the user choose whether to retrain the model or just run test case
    choice = input("Select Y for training, N for test (Y/N): ").strip().upper()

    # Training and Test selected
    if choice == 'Y':
        # save best loss values for best weights update 
        best_val_loss = float('inf')
        if os.path.exists(weigths_path):
            checkpoint = torch.load(weigths_path, map_location=hardware)
            if isinstance(checkpoint, dict) and 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                print("best validation loss was found from old runs")
        # pretrained model training loop
        num_epochs = 10
        count = 0
        for epoch in range(num_epochs):
            count = count + 1
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
            total = 0
            correct = 0
            val_loss = 0.0  
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(hardware), labels.to(hardware)
                    outputs = pretrained_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                avg_val_loss = val_loss/len(valid_loader)
                accuracy = 100*correct/total
                print(f"Epoch N{count} Validation accuracy: {accuracy}, Validation loss: {avg_val_loss}, Best Saved Loss: {best_val_loss}")
            
            # saving weights if the loss is better than saved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'model_state_dict': pretrained_model.state_dict(),
                    'best_val_loss': best_val_loss
                }
                torch.save(checkpoint, weigths_path)
                print("Saved weights were updated due to better loss value")    
        
        # time taken showcase
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Process took {elapsed_time:.2f} seconds")

    # Test only selected
    elif choice == 'N':
        print("Starting test phase")
        if os.path.exists(weigths_path):
            checkpoint = torch.load(weigths_path, map_location=hardware)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pretrained_model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded saved weights")
        else:
            raise FileNotFoundError("No weights found, you must train model first!")
    else: 
        raise ValueError("Invalid selection!")

    # test phase
    test_and_visalize(model=pretrained_model, test_loader=test_loader, test_dataset=test_dataset, hardware=hardware, class_count=class_count, model_name="ResNet18_Frozen")