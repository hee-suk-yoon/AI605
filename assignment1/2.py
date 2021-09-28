from datasets import load_dataset
from pprint import pprint
from torch import nn 
import torch
from torch.utils.data import DataLoader,Dataset
sst_dataset = load_dataset('sst')

#RNN implementation


#check GPU
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    


#Problem 2.2
vocab_dict = {}
for idx_, text in enumerate(sst_dataset['train']['sentence']):
    space_tokenized = text.split(' ')
    for idx2_, word in enumerate(space_tokenized):
        if not(word in vocab_dict):
            vocab_dict[word] = 1
        else:
            vocab_dict[word] += 1

#remove words that did not occur at least 2 timescl
vocab = [k for k,v in vocab_dict.items() if v >= 2]
vocab = ['PAD', 'UNK'] + vocab 
word2id = {word: id_ for id_, word in enumerate(vocab)}
print(len(vocab))


def create_dataloader(data,length):
    #convert all training sentence with its word2id 
    input = [] 
    for idx_, text in enumerate(data['sentence']):
        space_tokenized = text.split(' ')
        for idx_, word in enumerate(space_tokenized):
            if word in word2id:
                space_tokenized[idx_] = word2id[word]
            else: 
                space_tokenized[idx_] = word2id['UNK']
        #pad or truncate
        if len(space_tokenized) < length:
            space_tokenized = space_tokenized + [word2id['PAD']] * (length - len(space_tokenized)) # PAD tokens at the end
        else:
            space_tokenized = space_tokenized[:length]
        input.append(space_tokenized)

    #create DataLoader
    train = torch.utils.data.TensorDataset(torch.Tensor(input).to(torch.int32), torch.round(torch.Tensor(data['label'])).to(torch.long))
    return DataLoader(train, batch_size = 16, shuffle=False)

length = 16
train_loader = create_dataloader(sst_dataset['train'],length)
val_loader = create_dataloader(sst_dataset['validation'],length)
test_loader = create_dataloader(sst_dataset['test'],length)



#Vanilla RNN 
class Model(nn.Module):
    def __init__(self, d, length):
        super(Model, self).__init__()
        #input-hidden weight (Embedding matrix)
        self.hidden_dim = d
        self.U = nn.Embedding(len(vocab), self.hidden_dim)

        #hidden-hidden weight
        self.V = nn.Linear(self.hidden_dim, self.hidden_dim)

        #hidden-output weight
        self.W = nn.Linear(self.hidden_dim, 2)  

        self.m = nn.Softmax(dim=1)
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        #Iterate through sentence
        for i, word_id in enumerate(torch.transpose(x,0,1)):
            word_id = word_id.view(batch_size,-1) #shape (batch_size, 1)
            emb = self.U(word_id) #shape (batch_size, 1, self.hidden_dim)
            emb_flat = emb.view(emb.size(0), -1) # shape (batch_size, 1*self.hidden_dim)
            hidden = torch.tanh(emb_flat+self.V(hidden)) # shape (batch_size, self.hidden_dim)
        out = self.W(hidden)

        return self.m(out)
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        return hidden

#create model obejct
d = 128
rnn_model = Model(d, length).to(device)
cel = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(rnn_model.parameters(), lr = 0.01)


train_loss_value = []
val_loss_value = []
for epoch in range(0,100):
    train_loss = 0
    for batch_id, (data,label) in enumerate(train_loader):
        data=data.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        out = rnn_model(data)
        loss = cel(out,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average training loss: {:.4f}'.format(
          epoch, average_loss))
    train_loss_value.append(average_loss)
    
    
    val_loss = 0
    for batch_id, (data,label) in enumerate(val_loader):
        data=data.to(device)
        label=label.to(device)
        out = rnn_model(data)
        loss = cel(out,label)
        val_loss += loss.item()
    average_loss = val_loss / len(val_loader.dataset)
    print('====> Epoch: {} Average validation loss: {:.4f}'.format(
          epoch, average_loss))
    val_loss_value.append(average_loss)

import numpy as np
np.save('RNN_train_loss_value.npy', np.array(train_loss_value))
np.save('RNN_val_loss_value.npy', np.array(val_loss_value))

