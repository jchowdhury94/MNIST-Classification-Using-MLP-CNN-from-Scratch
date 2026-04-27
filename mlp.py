import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import random

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)



def sigmoid(x):  

    return 1 / (1 + np.exp(-x))

def softmax(x):  

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ===================== Data Loading ===================== #

def dataloader(train_dataset, test_dataset, batch_size=128):

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform
    )

    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))

    return dataloader(train_dataset, test_dataset)



class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.lr = lr

        
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        outputs = softmax(self.z2)

        return outputs

    def backward(self, x, y, pred):
        batch_size = x.shape[0]


        y_one_hot = np.zeros((batch_size, pred.shape[1]))
        y_one_hot[np.arange(batch_size), y] = 1

        dz2 = pred - y_one_hot

        dw2 = np.dot(self.a1.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size


        dz1 = np.dot(dz2, self.w2.T) * self.a1 * (1 - self.a1)


        dw1 = np.dot(x.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size


        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def cross_entropy_loss(self, pred, y):

        batch_size = y.shape[0]
        correct_probs = pred[np.arange(batch_size), y]
        loss = -np.mean(np.log(correct_probs + 1e-9))  # avoid log(0)
        return loss

    def train(self, x, y):

        pred = self.forward(x)


        loss = self.cross_entropy_loss(pred, y)


        self.backward(x, y, pred)

        return loss



def main():
    
    train_loader, test_loader = load_data()


    input_size = 28 * 28
    hidden_size = 256
    output_size = 10
    lr = 0.1
    num_epochs = 30

    
    model = MLP(input_size, hidden_size, output_size, lr)

    
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:
            
            x = inputs.view(-1, input_size).numpy()
            y = labels.numpy()

            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

    
    correct_pred = 0
    total_pred = 0

    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()

        pred = model.forward(x)
        predicted_labels = np.argmax(pred, axis=1)

        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)

    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":
    main()
