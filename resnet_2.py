"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

outputs_print=[]
labels_print=[]

class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags
        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        
        #other convolutions needed for the residuals
        self.conv1a = nn.Conv2d(3, self.num_channels*4, 3, stride = 1, padding = 1)
        
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        
        #other convoluations needed for the residuals
        self.conv2a = nn.Conv2d(self.num_channels*2, self.num_channels*2, 3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)
        
        #other convolutions needed for the residuals
        self.conv4 = nn.Conv2d(self.num_channels*4, self.num_channels*8, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.num_channels*8)
        self.conv4a = nn.Conv2d(self.num_channels*4, self.num_channels*10, 3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(self.num_channels*10)
        self.conv5 = nn.Conv2d(self.num_channels*8, self.num_channels*10, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(self.num_channels*10)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(8*8*self.num_channels*10, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 11) #changing from 6 in order to fit our project        
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .
        Returns:
            out: (Variable) dimension batch_size x 10 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 64 x 64
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        residual = s
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
        s = F.relu(s)                                       # batch_size x num_channels x 64 x 64
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 64 x 64
        s = F.relu(s)                                       # batch_size x num_channels*2 x 64 x 64
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 64 x 64
        residual = self.bn3(self.conv1a(residual))
        s = F.relu(F.max_pool2d(s + residual, 2))           # batch_size x num_channels*4 x 32 x 32
        
        residual = s
        s = self.bn4(self.conv4(s))
        s = F.relu(F.max_pool2d(s, 2))                     # batch_size x num_channels*4 x 16 x 16
        s = self.bn5(self.conv5(s))
        
        residual=self.bn4a(self.conv4a(residual))
        residual=F.max_pool2d(residual, 2)
        s = F.relu(F.max_pool2d(s + residual, 2))           # batch_size x num_channels*4 x 8 x 8
        
        # flatten the output for each image
        s = s.view(-1, 8*8*self.num_channels*10)             # batch_size x 8*8*num_channels*10

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 6 <-This needs to be 11!

        # apply log softmax on each image's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 11 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5,..., 10]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5, 11]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

#adding F1 Score
def f1_score(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    tags=set(labels)
    predicted=set(outputs)

    tp=len(tags.intersection(outputs))
    fp=len(predicted.difference(labels))
    fn=len(tags.difference(outputs))

    if tp>0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)


        return 2*((precision*recall)/(precision+recall))
    else:
        return 0

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy, 'f1_score' : f1_score,
    # could add more metrics such as accuracy for each token type
}
