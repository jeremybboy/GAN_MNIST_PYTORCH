import torch.nn as nn

class CNNDiscriminator(nn.Module):
    def __init__(self):
        super(CNNDiscriminator, self).__init__()
        self.conv1=nn.Conv2d(1, 32, 4, 2, 1, bias=False)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(32, 32*2, 4, 2, 1, bias=False)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(32*2, 32*4, 3, 2, 1, bias=False)
        self.relu3=nn.ReLU()
        self.conv4=nn.Conv2d(32*4, 1, 4, 1, 0, bias=False)
        self.sig=nn.Sigmoid()

    def forward(self, x):
        out=self.conv1(x)
        out=self.relu1(out)
        out=self.conv2(out)
        out=self.relu2(out)
        out=self.conv3(out)
        out=self.relu3(out)
        out=self.conv4(out)
        out=self.sig(out)
        return out.view(-1, 1)

