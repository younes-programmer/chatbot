import random
import json
import numpy as np
import torch
import torch as nn
from torch.utils.data import Dataset, DataLoader
from pre_processing import *
from model import *
intents = json.loads(open('intents.json').read())


tags = []
all_words = []
documents =[]
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word_list = tokenize(pattern)
        all_words.extend(word_list)
        documents.append((word_list, tag))

ignore_words = ['?', '!', '.', '#', '%', '@', '*']
# stem all words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# test is steaming is working => print(all_words)
X_train = []
Y_train = []
for (pattern_sentence, tag) in documents:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(len(X_train))
# print(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.size_of_simple = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.size_of_simple



# Hyperparameters
batch_size = 8
shuffle = True
number_of_workers = 2
output_size = len(tags)
input_size = len(X_train[0])
hidden_size = 8
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=number_of_workers)

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size)

# loss and optimizer
entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = entropy_loss(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch +1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# dictionary
data = {
    "model-state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
my_file = "data.pth"
torch.save(data, my_file)

print(f"training is complete !!")
