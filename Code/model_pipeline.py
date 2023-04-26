import torch
from transformers import GPT2Model
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
### Define Dataset Class Function
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, labels, tokenizer):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt") for text in df['text']]
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Get a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Get a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_labels= self.get_batch_labels(idx)
        return batch_texts, batch_labels
    
### Create Trainer function #################################################################################
def train(model, model_name, train_data, val_data, labels, tokenizer, learning_rate, epochs):
    train, val = Dataset(train_data, labels, tokenizer), Dataset(val_data, labels, tokenizer)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if model_name == 'gpt2': 
        criterion = nn.CrossEntropyLoss()
    
    if use_cuda:
        model = model.cuda()
        if model_name == 'gpt2':
            criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            
            input_id = train_input["input_ids"].squeeze(1).to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            
            model.zero_grad()
            
            if model_name =='gpt2':
                output = model(input_id, mask)
            else:
                output = model(input_id, mask, labels=train_label)
            
            if model_name == 'gpt2':
                batch_loss = criterion(output, train_label)
            else:
                batch_loss = output.loss 
                
            total_loss_train += batch_loss.item()
            
            if model_name == 'gpt2':
                acc = (output.argmax(dim=1)==train_label).sum().item()
            else:    
                acc = (output.logits.argmax(dim=1)==train_label).sum().item()
            
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        
        with torch.no_grad():
            
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                mask = val_input['attention_mask'].squeeze(1).to(device)
                
                if model_name =='gpt2':
                    output = model(input_id, mask)
                else:
                    output = model(input_id, mask, labels=val_label)
                
                if model_name =='gpt2':
                    batch_loss = criterion(output, val_label)
                else:
                    batch_loss = output.loss
                total_loss_val += batch_loss.item()
                
                if model_name == 'gpt2':
                    acc = (output.argmax(dim=1)==val_label).sum().item() 
                else:
                    acc = (output.logits.argmax(dim=1)==val_label).sum().item()
                total_acc_val += acc
                
            print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train/len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}")

#### Model evaluator function ##############################################################################

def evaluate(model, model_name, test_data, labels, tokenizer):

    test = Dataset(test_data, labels, tokenizer)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        print(f'GPU NAME: {torch.cuda.get_device_name()}')
        model = model.cuda()

        
    # Tracking variables
    predictions_labels = []
    true_labels = []
    
    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_input['attention_mask'].squeeze(1).to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            
            if model_name =='gpt2':
                output = model(input_id, mask)
            else:
                output = model(input_id, mask, labels=test_label)
                
            if model_name == 'gpt2':
                acc = (output.argmax(dim=1) == test_label).sum().item()
            else:
                acc = (output.logits.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
            
            # add original labels
            true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.logits.argmax(dim=1).cpu().numpy().flatten().tolist()
    
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    return true_labels, predictions_labels

#############################################################################################################

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output