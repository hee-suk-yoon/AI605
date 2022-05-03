# dfdldkfldkf
from datasets import load_dataset
from pprint import pprint
from torch import nn 
import torch
from torch.utils.data import DataLoader,Dataset
sst_dataset = load_dataset('sst')
#pprint(sst_dataset['train']['sentence'])

#Problem 2.1
#Sapce tokenization
vocab = ['PAD', 'UNK']
for idx_, text in enumerate(sst_dataset['train']['sentence']):
    space_tokenized = text.split(' ')
    for idx2_, word in enumerate(space_tokenized):
        if not(word in vocab):
            vocab.append(word)
        


word2id = {word: id_ for id_, word in enumerate(vocab)}
print(len(vocab))



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


#Problem 3
# Two-layer MLP classification
class Baseline(nn.Module):
  def __init__(self, d, length):
    super(Baseline, self).__init__()
    self.embedding = nn.Embedding(len(vocab), d)
    self.layer = nn.Linear(d * length, d, bias=True)
    self.relu = nn.ReLU()
    self.class_layer = nn.Linear(d, 2, bias=True)

  def forward(self, input_tensor):
    emb = self.embedding(input_tensor) # [batch_size, length, d]
    emb_flat = emb.view(emb.size(0), -1) # [batch_size, length*d]
    hidden = self.relu(self.layer(emb_flat))
    logits = self.class_layer(hidden)
    return logits

#convert all training sentence with its word2id 
length = 16
training_data = [] 
for idx_, text in enumerate(sst_dataset['train']['sentence']):
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
    training_data.append(space_tokenized)

#create DataLoader
train = torch.utils.data.TensorDataset(torch.Tensor(training_data).to(torch.int32), torch.round(torch.Tensor(sst_dataset['train']['label'])).to(torch.long))
train = DataLoader(train, batch_size = 16, shuffle=True)




#create model obejct
d = 128
baseline = Baseline(d, length)
softmax = nn.Softmax(1)
cel = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.SGD(baseline.parameters(), lr = 0.1)

for epoch in range(0,10):
    for idx, (data,label) in enumerate(train):
        optimizer.zero_grad()
        logits = baseline(data)
        loss = cel(logits,label)
        loss.backward()
        optimizer.step
