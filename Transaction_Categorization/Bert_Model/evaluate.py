from data_processing import *
import torch
import numpy as np    
from sklearn.metrics import f1_score, accuracy_score

## Calculate the F1 score for a multi-class classification task.
## Args: preds-Predicted labels,  labels-True labels
def f1_func(preds, labels):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()
    f1 = f1_score(l, p, average='weighted')
    return f1

## Calculate and print accuracy for each class
## Calculate and print overall accuracy score
## Args: preds-Predicted labels,  labels-True labels, lab_dict_inverse-Inverse label dictionary
def accuracy_per_class(preds, labels, label_dict_inverse):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()

    class_accuracies = {}
    for label in np.unique(l):
        mask = l == label
        y_preds = p[mask]
        y_true = l[mask]
        class_name = label_dict_inverse[label]
        class_accuracy = accuracy_score(y_true, y_preds)
        class_accuracies[class_name] = class_accuracy

    overall_accuracy = accuracy_score(l, p)

    # Print class accuracies
    for class_name, class_accuracy in class_accuracies.items():
        print(f'Class: {class_name}\nAccuracy: {class_accuracy:.2%}\n')

    # Print overall accuracy
    print(f'Overall Accuracy: {overall_accuracy:.2%}')

def evaluate(dataloader_val, model):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total/len(dataloader_val)

    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
