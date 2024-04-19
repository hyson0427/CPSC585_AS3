import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as AF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

###################### Designing an ANN architectures #########################

### CNN architecture
class CNN(nn.Module):
    # The probability of dropout, number of hidden nodes, number of output classes
    def __init__(self, dropout_pr, num_hidden, num_classes):
        super(CNN, self).__init__()

        self.pool_size = 2
        self.filter_size = 5 # 1, 5 tested
        self.stride = 1
        self.output_size = ((28 - self.filter_size) / self.stride ) + 1
        self.w = ((self.output_size - self.pool_size) / self.pool_size) + 1
        self.num_flatten_nodes = int(num_classes * self.w * self.w)
        print('output_size : ', self.output_size, 'w : ', self.w, '  num_flatten_nodes: ', self.num_flatten_nodes)

        # convolutional layer = 2
        self.conv1 = nn.Conv2d(1, 10, self.filter_size) # input_channel = 1, filter = 10
        self.conv2 = nn.Conv2d(10, 10, self.filter_size)

        self.drop2D = nn.Dropout2d(p=0.25, inplace=False)
        self.mp = nn.MaxPool2d(self.pool_size)
        #self.mp = nn.AvgPool2d(self.pool_size)
        #self.fc1 = nn.Linear(160, 10) # 4*4*10 Vector

        self.fc1 = nn.Linear(160, 100)
        self.fc2 = nn.Linear(100, 10) # to 10 outputs

    def forward(self, x):
      x = AF.relu(self.mp(self.conv1(x)))
      x = AF.relu(self.mp(self.conv2(x)))
      x = self.drop2D(x)
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = self.fc2(x)
      return AF.log_softmax(x)

def show_some_digit_images(images):
    print("> Shapes of image:", images.shape)
    for i in range(0, 10):
        plt.subplot(2, 5, i+1) # Display each image at i+1 location in 2 rows and 5 columns (total 2*5=10 images)
        plt.imshow(images[i][0], cmap='Oranges') # show ith image from image matrices by color map='Oranges'
    plt.show()

# Training function
def train_ANN_model(num_epochs, training_data, device, CUDA_enabled, is_MLP, ANN_model, loss_func, optimizer):
    train_losses = []
    ANN_model.train() # to set the model in training mode. Only Dropout and BatchNorm care about this flag.
    for epoch_cnt in range(num_epochs):
        for batch_cnt, (images, labels) in enumerate(training_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            optimizer.zero_grad() # set the cumulated gradient to zero
            output = ANN_model(images) # feedforward images as input to the networkf
            loss = loss_func(output, labels) # computing loss
            train_losses.append(loss.item())
            loss.backward() # calculating gradients backward using Autograd
            optimizer.step() # updating all parameters after every iteration through backpropagation

            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"Epoch={epoch_cnt+1}/{num_epochs}, batch={batch_cnt+1}/{num_train_batches}, loss={loss.item()}")
    return train_losses

# Testing function
def test_ANN_model(device, CUDA_enabled, is_MLP, ANN_model, testing_data):
    predicted_digits=[]
    with torch.no_grad():
        ANN_model.eval() # # set the model in testing mode. Only Dropout and BatchNorm care about this flag.
        for batch_cnt, (images, labels) in enumerate(testing_data):
            if (is_MLP):
                images = images.reshape(-1, 784) # or images.view(-1, 784) or torch.flatten(images, start_dim=1)

            if (device.type == 'cuda' and CUDA_enabled):
                images = images.to(device) # moving tensors to device
                labels = labels.to(device)

            output = ANN_model(images)
            _, prediction = torch.max(output,1) # returns the max value of all elements in the input tensor
            predicted_digits.append(prediction)
            num_samples = labels.shape[0]
            num_correct = (prediction==labels).sum().item()
            accuracy = num_correct/num_samples
            if (batch_cnt+1) % mini_batch_size == 0:
                print(f"batch={batch_cnt+1}/{num_test_batches}")
        print("> Number of samples=", num_samples, "number of correct prediction=", num_correct, "accuracy=", accuracy)
    return predicted_digits

########################### Checking GPU and setup #########################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if (torch.cuda.is_available()):
    print("The CUDA version is", torch.version.cuda)
    # Device configuration: use GPU if available, or use CPU
    cuda_id = torch.cuda.current_device()
    print("ID of the CUDA device:", cuda_id)
    print("The name of the CUDA device:", torch.cuda.get_device_name(cuda_id))
    print("GPU will be utilized for computation.")
else:
    print("CUDA is supported in your machine. Only CPU will be used for computation.")
#exit()

############################### ANN modeling #################################
print("------------------ANN modeling---------------------------")
transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])

from torch.autograd import Variable
x = Variable

train_dataset=datasets.MNIST(root='./data', train=True, transform=transforms, download=True)
test_dataset=datasets.MNIST(root='./data', train=False, transform=transforms, download=False)
print("> Shape of training data:", train_dataset.data.shape)
print("> Shape of testing data:", test_dataset.data.shape)
print("> Classes:", train_dataset.classes)

train_dataloader=DataLoader(dataset=train_dataset, batch_size=mini_batch_size, shuffle=True)
test_dataloader=DataLoader(dataset=test_dataset, batch_size=mini_batch_size, shuffle=True)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)
print("> Mini batch size: ", mini_batch_size)
print("> Number of batches loaded for training: ", num_train_batches)
print("> Number of batches loaded for testing: ", num_test_batches)

iterable_batches = iter(train_dataloader) # making a dataset iterable
images, labels = next(iterable_batches) # If you can call next() again, you get the next batch until no more batch left
show_digit_image = True
if show_digit_image:
    show_some_digit_images(images)

num_input = 28*28   # 28X28=784 pixels of image
num_classes_mlp = 18    # output layer
num_hidden_mlp = 12     # number of neurons at the first hidden layer

num_classes_cnn = 10 # output layer of CNN model
num_hidden_cnn = 20 # number of neurons at the first hidden layer

dropout_pr = 0.01

# CNN model
CNN_model = CNN(dropout_pr, num_hidden_cnn, num_classes_cnn)
print("> CNN model parameters")
print(CNN_model.parameters)

# To turn on/off CUDA if I don't want to use it.
CUDA_enabled = True
if (device.type == 'cuda' and CUDA_enabled):
    print("...Modeling using GPU...")
    #MLP_model = MLP_model.to(device=device) # sending to whaever device (for GPU acceleration)
    CNN_model = CNN_model.to(device=device)
else:
    print("...Modeling using CPU...")

loss_func = nn.CrossEntropyLoss() # for MLP
loss_func_cnn = nn.NLLLoss() # for CNN

num_epochs = 1
alpha = 0.02       # learning rate for MLP
gamma = 0.3       # momentum for MLP

gamma_cnn = 0.3       # momentum for CNN

CNN_optimizer = optim.Adagrad(CNN_model.parameters(), lr=alpha)

print("............Training CNN................")
is_MLP = False
train_loss=train_ANN_model(num_epochs, train_dataloader, device, CUDA_enabled, is_MLP, CNN_model, loss_func_cnn, CNN_optimizer)
print("............Testing CNN model................")
predicted_digits=test_ANN_model(device, CUDA_enabled, is_MLP, CNN_model, test_dataloader)

torch.save(CNN_model.state_dict(), "mnist_cnn.pt")
