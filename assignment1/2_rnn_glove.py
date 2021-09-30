from datasets import load_dataset
from pprint import pprint
from torch import nn 
import torch
from torch.utils.data import DataLoader,Dataset
from torchtext.vocab import GloVe
import numpy as np

sst_dataset = load_dataset('sst')
glove = GloVe(name='6B',dim = 300)
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
    
length = 16
data_train = sst_dataset['train']['sentence']
label_train = sst_dataset['train']['label']
data_test = sst_dataset['validation']

def create_dataloader(data, train):
    emb = torch.empty((len(data['sentence']),length,300),dtype = torch.float32)
    for idx_,text in enumerate(data['sentence']):
        space_tokenized = text.lower().split(' ')
        if len(space_tokenized) < length:
            padding = ['PAD'] * (length - len(space_tokenized))
            #space_tokenized = space_tokenized + padding
            space_tokenized = padding + space_tokenized
        else: 
            space_tokenized = space_tokenized[:length]

        for idx2_, word in enumerate(space_tokenized):
            emb[idx_][idx2_] = glove[word]

    data_torch = torch.utils.data.TensorDataset(emb, torch.round(torch.Tensor(data['label'])).to(torch.long))
    if train == 0:
        train_data, val_data = torch.utils.data.random_split(data_torch,[int(0.95*len(data_torch)),len(data_torch)-int(0.95*len(data_torch))], generator= torch.Generator().manual_seed(42) )
        return DataLoader(train_data, batch_size = 16, shuffle=True, drop_last = True), DataLoader(val_data, batch_size = 16, shuffle=True, drop_last = True)     
    else:
        return DataLoader(data_torch, batch_size = 16, shuffle=True, drop_last = True) 

"""
train = torch.utils.data.TensorDataset(train_emb, torch.round(torch.Tensor(label_train)).to(torch.long))
train_data, val_data = torch.utils.data.random_split(train,[int(0.95*len(train)),len(train)-int(0.95*len(train))], generator= torch.Generator().manual_seed(42) )
train_loader = DataLoader(train_data, batch_size = 16, shuffle=True, drop_last = True)
val_loader = DataLoader(val_data, batch_size = 16, shuffle=True, drop_last = True) 
"""
train_loader,val_loader = create_dataloader(sst_dataset['train'],0)
test_loader = create_dataloader(sst_dataset['validation'],1)



#Vanilla RNN 
class Model(nn.Module):
    def __init__(self, d):
        super(Model, self).__init__()
        #input-hidden weight (Embedding matrix)
        self.hidden_dim = d

        self.U = nn.Linear(self.hidden_dim,self.hidden_dim)

        #hidden-hidden weight
        self.V = nn.Linear(self.hidden_dim, self.hidden_dim)

        #hidden-output weight
        self.W = nn.Linear(self.hidden_dim, 2)  

        #m = nn.Softmax(dim=1)
    def forward(self, x):       #x: size [BatchSize, Length]

        # Initializing hidden state for first input using method defined below
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)


        #Iterate through sentence and input to RNN sequentially
        for t in range(0,length):
            xt = x[:,t,:] #shape [BatchSzie, D]
            hidden = torch.tanh(self.U(xt) + self.V(hidden))

        #output logit using last sequence
        return self.W(hidden)

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        return hidden


#create model obejct
d = 300
rnn_model = Model(d).to(device)
m = nn.Softmax(1)
cel = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(rnn_model.parameters(), lr = 0.01)


train_loss_value = []
val_loss_value = []
for epoch in range(0,15):
    train_loss = 0
    for batch_id, (data,label) in enumerate(train_loader):
        #print(label)
        data=data.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        out = rnn_model(data)
        #print(out)
        loss = cel(out,label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    average_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average training loss: {:.4f}'.format(
          epoch, average_loss))
    train_loss_value.append(average_loss)
    
    with torch.no_grad():
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




def accuracy(pred,target):
    correct = 0
    for i in range(0,pred.size()[0]):
        if pred[i] == target[i]:
            correct += 1
    return correct

total_correct = 0
total = 0
for batch_id, (data,label) in enumerate(test_loader):
    data = data.to(device)
    label = label.to(device)
    logits = rnn_model(data)
    pred = m(logits)
    print(pred)
    print(label)
    pred = torch.argmax(pred,dim=1)
    total_correct += accuracy(pred,label)
    total += label.size()[0]

print(total_correct/total)
