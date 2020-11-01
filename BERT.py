#!/usr/bin/env python
# coding: utf-8

# Section 1: Data cleaning

# In[69]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[3]:


df= pd.read_csv(
    'C:/Users/benha/Desktop/MasterProject/smileannotationsfinal.csv',
    names = ['id','text','category']
)
df.set_index('id', inplace=True)


# In[4]:


df.head()
df.text.iloc[0]


# In[5]:


# count the number of unique categories
df.category.value_counts()


# In[6]:


df = df[~df.category.str.contains('\|')]


# In[7]:


df = df[df.category != 'nocode']


# In[8]:


df.category.value_counts()


# In[9]:


possible_labels = df.category.unique()


# In[10]:


label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index


# In[11]:


label_dict


# In[12]:


df['label']=df.category.replace(label_dict)
df.head()


# Section 2: Training/Validation Split

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_val, Y_train, Y_val = train_test_split(
    df.index.values,
    df.label.values,
    test_size=0.15, #15% for validation
    random_state=17,
    stratify=df.label.values
)


# In[15]:


df['data_type']= ['not_set']*df.shape[0]
df.head()


# In[16]:


df.loc[X_train, 'data_type']='train'
df.loc[X_val, 'data_type']='val'
df.groupby(['category','label','data_type']).count()


# Section 3: Loading tokenizer and coding data

# In[17]:


get_ipython().system('pip install transformers==3.4.0')


# In[27]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[28]:


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)


# In[34]:


encoded_data_train = tokenizer.batch_encode_plus (
    df[df.data_type=='train'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)
#batch_encode_plus


# In[36]:



encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train= encoded_data_train['attention_mask']
labels_train= torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val= encoded_data_val['attention_mask']
labels_val= torch.tensor(df[df.data_type=='val'].label.values)


# In[37]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


# In[38]:


len(dataset_train)


# In[39]:


len(dataset_val)


# In[40]:


from transformers import BertForSequenceClassification


# In[41]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)


# In[43]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[44]:


batch_size = 32 # can be smaller or larger. Smaller: for low performance computers
dataloader_train = DataLoader(
    dataset_train,
    sampler = RandomSampler(dataset_train),
    batch_size = batch_size
)

dataloader_val = DataLoader(
    dataset_val,
    sampler = RandomSampler(dataset_val),
    batch_size = batch_size
)


# In[45]:


from transformers import AdamW, get_linear_schedule_with_warmup


# In[48]:


optimizer = AdamW(
    model.parameters(),
    lr= 1e-5, # recommeded from 2e-5 . 5e-5
    eps= 1e-8  # default
)


# In[49]:


epochs = 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train)*epochs
)


# In[50]:


import numpy as np


# In[53]:


from sklearn.metrics import f1_score


# In[ ]:


# preds = [0.9 0.05 0.05 0 0 0]
# preds = [1 0 0 0 0 0]


# In[54]:


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')


# In[58]:


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
            y_preds = pred_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


# In[59]:


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[60]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)


# In[81]:


def evaluate(dataloader_val):
    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],          
                }
        with torch.no_grad():
            outputs= model(**inputs)
        
        loss = outputs[0]
        logits= outputs[1]
        loss_val_total +=loss.item()
        
        logits= logits.detach().cpu().numpy()
        label_ids= inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg = loss_val_total/len(dataloader_val)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals


# In[ ]:


for epoch in tqdm(range(1, epochs+1)):
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch),
                        leave = False,
                        disable = False
                       )
    for batch in progress_bar:
        
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]
        }
        
        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total +=loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
        
        torch.save(model.state_dict(), f'BERT_ft_epoch{epoch}.model')
        
        tqdm.write('\nEpoch {epoch}')
        
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        val_loss, predictions, true_vals = evaluate(dataloader_val)
        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation_loss: {val_loss}')
        tqdm.write(f'F1 Score (weighted): {val_f1}')


# In[ ]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)


# In[ ]:


model.to(device)
pass


# In[ ]:


model.load_state_dict(torch.load('finetuned_bert_epoch_1_gpu_trained.model',
                                map.location = torch.device('cpu')))


# In[ ]:


_,predictions, true_vals = evaluate(dataloader_val)


# In[ ]:


accuracy_per_class(predictions, true_vals)


# In[ ]:




