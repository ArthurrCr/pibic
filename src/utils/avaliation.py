import matplotlib as plt
import torch
import numpy as np

def compute_iou(preds, labels, smooth=1e-6):
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def compute_metrics(preds, labels):
    preds = preds.view(-1)
    labels = labels.view(-1)
    
    TP = (preds * labels).sum().item()
    TN = ((1 - preds) * (1 - labels)).sum().item()
    FP = (preds * (1 - labels)).sum().item()
    FN = ((1 - preds) * labels).sum().item()
    
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return accuracy, precision, recall, f1

def visualize_predictions(images, masks, outputs):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = outputs.cpu().numpy()
    mean = [0.29697557, 0.29690879, 0.29623264]
    std = [0.19810056, 0.19796892, 0.19816074]
    for i in range(len(images)):
        image = images[i].transpose(1, 2, 0)
        image = (image * np.array(std).reshape(1, 1, 3)) + np.array(mean).reshape(1, 1, 3)
        image = np.clip(image, 0, 1)
        
        mask = masks[i][0]
        output = outputs[i]
        output = torch.sigmoid(torch.tensor(output)).numpy()
        pred = (output > 0.5).astype(np.uint8)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Imagem')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Máscara Real')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap='gray')
        plt.title('Máscara Predita')
        plt.axis('off')
        
        plt.show()
