import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

dataset = pd.read_csv('iris.csv')


mappings = {
    "Setosa" : 0,
    "Versicolor" : 1,
    "Virginica" : 2
}
dataset["variety"] = dataset["variety"].apply(lambda x : mappings[x])

X = dataset.drop("variety", axis=1).values
y = dataset["variety"].values

X = torch.FloatTensor(X)
y = torch.LongTensor(y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(4, 25)
        self.a1 = nn.ReLU()
        self.h2 = nn.Linear(25, 30)
        self.a2 = nn.ReLU()
        self.out = nn.Linear(30, 3)
       
    def forward(self, x):
        x = self.a1(self.h1(x))
        x = self.a2(self.h2(x))
        x = self.out(x)
        return x

model = Model()
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
losses = []

for i in range(epochs):

    y_pred = model.forward(X)
    loss = loss_fn(y_pred, y)

    losses.append(loss.item())
    print(f'epoch: {i:2}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# plt.plot(range(epochs), losses)
# plt.ylabel('Loss')
# plt.xlabel('epoch')
# plt.show()

predNum = 0
predY = 0
with torch.no_grad():
    for a, b in zip(X, y):
        pred = model(a).argmax().item()
        print(a, b.item(), end=" ")
        if b != pred:
            print(pred, end=" ")
            print('fail')
        else:
            predY += 1
            print(pred)
        predNum += 1

accuracy = (predY / predNum) * 100

print(f"Predictions: {predNum}")
print(f"Correct: {predY}")
print(f"Fails: {predNum - predY}")
print(f"Accuracy: {round(accuracy, 2)}%")



