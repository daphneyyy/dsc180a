# import required packages
from data_processing import *
from evaluate import *
import torch
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import classification_report


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def main():
    # Train the model

    for epoch in tqdm(range(1, epochs+1)):
        
        model.train()
        
        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:

            model.zero_grad()
            
            batch = tuple(b.to(device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }       

            outputs = model(**inputs)
            
            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        torch.save(model.state_dict(), 'saved_weights.pt')
            
        tqdm.write(f'\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)            
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_validation)
        val_f1 = f1_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')

    # Calculating the Accuracy per class and the overall Acurracy Score
    # Caculating the precision, recall, and f1-score
    _, predictions, true_vals = evaluate(dataloader_test)
    accuracy_per_class(predictions, true_vals, label_dict_inverse)

    preds = np.argmax(predictions, axis = 1)
    print(classification_report(labels_test, preds))

if __name__ == "__main__":
    main()
