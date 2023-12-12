# import required packages
from clean import *
import torch
from tqdm import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import random
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# upload the dataset and load the data.
# this dataset is the original dataset 
# and does not contain the dates and times.
file = 'Transacation_outflows_3k.pqt'
data = pd.read_parquet(file, engine='auto')

# Check if a GPU is available, and use it if possible, otherwise use the CPU
# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Filter the required categories and define a new dataset
# which only contains these categories.
categories_filter = ['GENERAL_MERCHANDISE', 'FOOD_AND_BEVERAGES', 'GROCERIES', 'TRAVEL', 'PETS', 'EDUCATION', 'OVERDRAFT', 'RENT', 'MORTGAGE']
data1 = data[data['category_description'].isin(categories_filter)]

# Only inlcude a subset of the dataset 
# to prevent running out of memory problem.
data2 = data1[:50000]

# Data Cleanning Process Part
## Changing memo_clean column values to all lower case first.
data2['memo_clean'] = data2['memo_clean'].str.lower()

# Applying thoese cleaning functions to the subset of the dataset
# that we choose.
print('---------- Cleaning the dataset... ----------')
data2['memo_clean'] = data2['memo_clean'].apply(clean_text1)
data2['memo_clean'] = data2['memo_clean'].apply(remove_key_phrases)
data2['memo_clean'] = data2['memo_clean'].apply(remove_special_char)
data2['memo_clean'] = data2['memo_clean'].apply(remove_xs)
data2['memo_clean'] = data2['memo_clean'].apply(standardize_phrase)
data2['memo_clean'] = data2['memo_clean'].apply(remove_multiple_spaces)
print('---------- Done cleaning. ----------')

# Check numbers of each categories.
data2['category_description'].value_counts()

# Assign labels to each categories.
labels = data2.category_description.unique()

label_dict = {}
for index, label in enumerate(labels):
    label_dict[label] = index

# Creating a label column for the dataset.
data2['label'] = data2.category_description.replace(label_dict)

# split dataset into train, validation and test sets using stratify.
train_text, temp_text, train_labels, temp_labels = train_test_split(data2['memo_clean'], data2['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=data2['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)
print('---------- Making Models ---------- ')
# Load the tokenizer from bert packages
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                        do_lower_case=True)

# Tokenize the text in all train, val and test datasets.
# Set the max_length to 256 for safe.
encoded_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


encoded_val = tokenizer.batch_encode_plus(
    val_text.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_test = tokenizer.batch_encode_plus(
    test_text.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

# Convert the tokenized list to tensors

input_ids_train = encoded_train['input_ids']
attention_masks_train = encoded_train['attention_mask']
labels_train = torch.tensor(train_labels.tolist())

input_ids_val = encoded_val['input_ids']
attention_masks_val = encoded_val['attention_mask']
labels_val = torch.tensor(val_labels.tolist())

input_ids_test = encoded_test['input_ids']
attention_masks_test = encoded_test['attention_mask']
labels_test = torch.tensor(test_labels.tolist())


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)

# Load the model and push to the device which we defined at the beginning.

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                    num_labels=len(label_dict),
                                                    output_attentions=False,
                                                    output_hidden_states=False)
model = model.to(device)

# Setting the batch size to three
# Using RandomSampler to randomly sample the training set.
# Using SequentialSampler for validation set to sequentially test the data.
# Using DataLoaer to improve efficient iteration and batching the data
# during training and validation.

batch_size = 30

dataloader_train = DataLoader(dataset_train, 
                            sampler=RandomSampler(dataset_train), 
                            batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                sampler=SequentialSampler(dataset_val), 
                                batch_size=batch_size)

dataloader_test = DataLoader(dataset_test, 
                                sampler=SequentialSampler(dataset_test), 
                                batch_size=batch_size)

# Define an optimizer
# Setting the epochs to be five
optimizer = AdamW(model.parameters(),
                lr=1e-5, 
                eps=1e-8)
                
epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)




# Define the performance metrics through F1_Score and Accuracy Score

label_dict_inverse = {v: k for k, v in label_dict.items()}
