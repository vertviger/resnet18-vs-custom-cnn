from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from PIL import Image, ImageOps
import time

def test_and_visalize(model, test_loader, test_dataset, hardware, class_count, model_name = "Model"):
    start_time = time.perf_counter()
    # test phase
    model.eval()
    all_predictions = []
    all_labels = []
    total = 0
    correct = 0

    class_worst_examples = {}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(hardware), labels.to(hardware)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (predicted == labels).sum().item()
            if len(class_worst_examples) < class_count:
                for i in range(labels.size(0)):
                    true_lbl = labels[i].item()
                    pred_lbl = predicted[i].item()
                    if true_lbl != pred_lbl:
                        if true_lbl not in class_worst_examples:
                            global_idx = batch_idx * test_loader.batch_size + i
                            img_path, _ = test_dataset.imgs[global_idx]
                            class_worst_examples[true_lbl] = {
                                'path': img_path, 
                                'pred': test_dataset.classes[pred_lbl]
                            }
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Test took {elapsed_time:.2f} seconds")
    
    accuracy = 100*correct/total
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # confusion matrix setup
    cm = confusion_matrix(all_labels, all_predictions)
    
    # individual class accuracies
    class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100
    
    # visualizing missclassified images
    fig, axes = plt.subplots(2, 4, figsize = (16,8))
    axes = axes.flatten()
    
    plt.style.use('dark_background') #for git if needed
    
    for class_idx, class_name in enumerate(test_dataset.classes):
        ax = axes[class_idx]
        class_acc = class_accuracies[class_idx]
        if class_idx in class_worst_examples:
            ex = class_worst_examples[class_idx]
            img = Image.open(ex['path'])
            img_padded = ImageOps.pad(img, (224, 224), color = (30, 30, 30))
            ax.imshow(img_padded)
            ax.set_title(f"Class: {class_name}\nAcc: {class_acc:.1f}\nMissclassified as: {ex['pred']}", color = 'green', fontsize = 9)
        else:
            ax.text(0.5, 0.5, f"No errors for\n{class_name}!", horizontalalignment='center', verticalalignment='center', fontsize=10)
            ax.set_title(f"Class: {class_name}\nAcc: {class_acc:.1f}%", color='green', fontsize=9)
            
        ax.axis('off')
    axes[7].axis('off')
    plt.suptitle(f'{model_name} per-class performance showcase')
    plt.tight_layout()
    plt.show()

    # confusion matrix display
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = test_dataset.classes)
    disp.plot(cmap = plt.cm.Greens, ax = ax)
    plt.title(f"Test confusion matrix, accuracy: {accuracy:.2f}%")
    plt.show()