import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch


np.random.seed(42)
torch.manual_seed(42)



def relu(x):
    
    return np.maximum(0, x)

def softmax(x):
    
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, y):
    
    batch_size = y.shape[0]
    correct_probs = pred[np.arange(batch_size), y]
    loss = -np.mean(np.log(correct_probs + 1e-9))  
    return loss




def dataloader(train_dataset, test_dataset, batch_size=64):
    
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

    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    return dataloader(train_dataset, test_dataset, batch_size=128)




class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size
        self.lr = lr

        self.conv_output_size = input_size - kernel_size + 1

        self.flatten_size = num_filters * self.conv_output_size * self.conv_output_size

        
        self.conv_kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.01
        self.conv_bias = np.zeros((num_filters,))

        self.fc_w = np.random.randn(self.flatten_size, fc_output_size) * 0.01
        self.fc_b = np.zeros((1, fc_output_size))

    def forward(self, x):
    
        batch_size = x.shape[0]

        
        self.x = x.reshape(batch_size, self.input_size, self.input_size)

        
        self.conv_out = np.zeros((
            batch_size,
            self.num_filters,
            self.conv_output_size,
            self.conv_output_size
        ))

        for n in range(batch_size):  
            for f in range(self.num_filters):  
                for i in range(self.conv_output_size):  
                    for j in range(self.conv_output_size):  
                        region = self.x[n, i:i+self.kernel_size, j:j+self.kernel_size]
                        self.conv_out[n, f, i, j] = np.sum(region * self.conv_kernels[f]) + self.conv_bias[f]

    
        self.relu_out = relu(self.conv_out)

        
        self.flatten_out = self.relu_out.reshape(batch_size, -1)

        
        self.fc_out = np.dot(self.flatten_out, self.fc_w) + self.fc_b

        
        outputs = softmax(self.fc_out)
        self.pred = outputs

        return outputs

    def backward(self, x, y, pred):
        
        batch_size = y.shape[0]

        
        y_one_hot = np.zeros((batch_size, self.fc_output_size))
        y_one_hot[np.arange(batch_size), y] = 1

        
        dz2 = (pred - y_one_hot) / batch_size

      
        dW_fc = np.dot(self.flatten_out.T, dz2)
        db_fc = np.sum(dz2, axis=0, keepdims=True)

     
        daflat = np.dot(dz2, self.fc_w.T)

        
        daconv = daflat.reshape(self.relu_out.shape)

        
        dzconv = daconv * (self.conv_out > 0)

    
        dK = np.zeros_like(self.conv_kernels)
        db_conv = np.zeros_like(self.conv_bias)

     
        for n in range(batch_size):
            for f in range(self.num_filters):
                db_conv[f] += np.sum(dzconv[n, f])
                for i in range(self.conv_output_size):
                    for j in range(self.conv_output_size):
                        region = self.x[n, i:i+self.kernel_size, j:j+self.kernel_size]
                        dK[f] += dzconv[n, f, i, j] * region

       
        dK /= batch_size
        db_conv /= batch_size

        
        self.fc_w -= self.lr * dW_fc
        self.fc_b -= self.lr * db_fc
        self.conv_kernels -= self.lr * dK
        self.conv_bias -= self.lr * db_conv

    def train(self, x, y):
        
        pred = self.forward(x)

        
        loss = cross_entropy_loss(pred, y)

        
        self.backward(x, y, pred)

        return loss




def main():
   
    train_loader, test_loader = load_data()

   
    input_size = 28
    num_epochs = 5
    num_filters = 1
    kernel_size = 5
    fc_output_size = 10
    lr = 0.01

    
    model = CNN(input_size, num_filters, kernel_size, fc_output_size, lr)

    print("Starting training...")

    
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"Epoch {epoch+1} started")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Convert to numpy and reshape
            x = inputs.squeeze(1).numpy()
            x = x.reshape(x.shape[0], -1)
            y = labels.numpy()

            loss = model.train(x, y)
            total_loss += loss

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete. Starting evaluation...")

    
    correct_pred = 0
    total_pred = 0

    for inputs, labels in test_loader:
        x = inputs.squeeze(1).numpy()
        x = x.reshape(x.shape[0], -1)
        y = labels.numpy()

        pred = model.forward(x)
        predicted_labels = np.argmax(pred, axis=1)

        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)

    print(f"Test Accuracy: {correct_pred / total_pred:.4f}")


if __name__ == "__main__":
    main()