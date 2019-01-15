
import torch
from torch import nn
from torch.nn import functional as F
class Generator(nn.Module):
    # initializers
    def __init__(self, z_size, d=128, channels=1):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(z_size, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, channels, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):#, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x)) * 0.5 + 0.5
        return x


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=1):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        logits = self.conv4(x)
        prob = torch.tanh(logits)
        return logits, prob 


class Encoder(nn.Module):
    # initializers
    def __init__(self, z_size, d=128, channels=1):
        super(Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(channels, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d // 2, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, z_size, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x

class PDense(nn.Module):
    # initializers
    def __init__(self, input_size, hidden_size, output_size):
        super(PDense, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method 
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        logits = self.linear2(x)
        prob = torch.softmax(logits, dim=1)
        return logits, prob


class Dense(nn.Module):
    # initializers
    def __init__(self, input_size, hidden_size, output_size, batch_size,
                 active_fn=F.sigmoid):
        super(Dense, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.active_fn=active_fn
    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method 
    def forward(self, x):
        x = F.leaky_relu(self.linear1(x), 0.2)
        logits = self.linear2(x)
        prob = self.active_fn(logits)
        return logits, prob

class ZDiscriminator(nn.Module):
    # initializers
    def __init__(self, z_size, batchSize, d=128):
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.dropout(F.leaky_relu((self.linear2(x)), 0.2))
        x = F.sigmoid(self.linear3(x))
        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
