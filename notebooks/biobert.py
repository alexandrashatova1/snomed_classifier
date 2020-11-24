import tensorflow as tf
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
from torch import nn, optim
from collections import OrderedDict, defaultdict

# Tell PyTorch to use the GPU.    
device = torch.device("cuda")
print('Using GPU:', torch.cuda.get_device_name(0))

print('Loading the dataset into a pandas dataframe.')
df = pd.read_csv('sam_train.csv')
df2 = df[['id_int', 'text']].copy()
sentences = df2.text.values
labels = df2.id_int.values

id_list = df2['id_int'].tolist()
unique_val=list(set(id_list))
num_labels = len(unique_val)

#### Preparing data ##### 
print('Tokenizer and input formatting')
tokenizer = BertTokenizer(vocab_file='biobert_v1.1_pubmed/vocab.txt', do_lower_case=False)

# check longest sentence length

max_len = 0
for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print('Max sentence length: ', max_len)

MAX_LEN = 64

# Tokenize all of the sentences and map the tokens to their word IDs.
input_ids = []
attention_masks = []
for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                    sent,                      
                    add_special_tokens = True, 
                    max_length = 64,           
                    padding='max_length',
                    return_token_type_ids=False,
                    return_attention_mask = True,   
                    return_tensors = 'pt',     
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

print('Creating train-validation split, and creating dataloaders')
dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


batch_size = 32
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )


validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )

#### Creating a model ####
print('Creating a model')

config = BertConfig.from_json_file('biobert_v1.1_pubmed/bert_config.json')
state_dict = torch.load('biobert_v1.1_pubmed/pytorch_model.bin', map_location=device)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if k.startswith('bert.'):
      k = k.replace('bert.', '')
      new_state_dict[k] = v
    elif k.startswith('cls.'):
        continue
    else:
        new_state_dict[k] = v

class BioBertNER(nn.Module):

  def __init__(self, num_labels, config, state_dict):
    super().__init__()
    self.bert = BertModel(config)
    self.bert.load_state_dict(state_dict, strict=False)
    self.dropout = nn.Dropout(p=0.3)
    self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    self.softmax = nn.Softmax(dim=1)
    
  def forward(self, input_ids, attention_mask):
    encoded_layer, pooled_output = self.bert(input_ids=input_ids,
      attention_mask=attention_mask)
    enlayer = encoded_layer[-1]
    enlayer = self.dropout(enlayer)
    outlayer = self.classifier(enlayer)
    pooled_output = self.dropout(pooled_output)
    out = self.classifier(pooled_output)
    return out, outlayer

model = BioBertNER(num_labels,config,new_state_dict)
model.to(device)
print(model)

# Prep for training
print('Training:')

epochs = 4
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
total_steps = len(train_dataloader) * epochs

max_grad_norm=1.0
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

print('Checking output layer:')
params = list(model.named_parameters())
for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

## TRAINING ####
print('Training loop')

# Defining train loop:
def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler):

  model.train()

  losses = []
  correct_predictions = 0
    
  for step,batch in enumerate(data_loader):
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_labels = batch

      outputs, logits = model(b_input_ids,b_input_mask)

      _,preds = torch.max(outputs,dim=1)
      loss = loss_fn(outputs,b_labels)

      correct_predictions += torch.sum(preds == b_labels)
      losses.append(loss.item())
    
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

      epoch_loss = np.mean(losses)
      epoch_acc = correct_predictions.double()/len(data_loader)

  return epoch_acc , epoch_loss


#Defining eval loop:

def eval_model(model, data_loader, loss_fn, device):
  model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
      for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
        
            outputs, logits= model(b_input_ids,b_input_mask)
        
            _,preds = torch.max(outputs,dim=1)
            loss = loss_fn(outputs,b_labels)
            correct_predictions += torch.sum(preds == b_labels)
            losses.append(loss.item())

            epoch_loss = np.mean(losses)
            epoch_acc = correct_predictions.double()/len(data_loader)

      return epoch_acc , epoch_loss

# Now training:

loss_values = []

for epoch in range(epochs):
  total_loss = 0

  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_dataloader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler
  )

  print(f'Training loss: {train_loss} Training accuracy: {train_acc}')

  total_loss += train_loss.item()
  avg_train_loss = total_loss / len(train_dataloader)  
  loss_values.append(avg_train_loss)


  val_acc, val_loss = eval_model(
    model,
    validation_dataloader,
    loss_fn, 
    device
  )

  print(f'Validation loss: {val_loss} Validation accuracy: {val_acc}')
  print()
