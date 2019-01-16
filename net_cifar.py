
import torch
from torch import nn
from torch.nn import functional as F
class Generator(nn.Module):
    # initializers
    def __init__(self, z_size, d=128, channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 4, 1, 0, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    #weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            # nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 1, 4, 1, 0, bias = False),
            # nn.Sigmoid()
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        logits = self.main(input)
        prob = torch.tanh(logits)
        return logits, prob

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

class Encoder(nn.Module):
    # initializers
    def __init__(self, z_size, d=128, channels=3):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            # nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, z_size, 4, 1, 0, bias = False),
            # nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    #     x = F.leaky_relu(self.conv1_1(input), 0.2)
    #     x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
    #     x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
    #     x = self.conv4(x)
    #     return x
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

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