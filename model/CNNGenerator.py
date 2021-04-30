import torch.nn as nn
import torch.nn.functional as F



class CNNGenerator(nn.Module):
    def __init__(self,z_dim):
        super(CNNGenerator, self).__init__()
        self.z_dim = z_dim
        self.convt1=nn.ConvTranspose2d(z_dim, 32 * 4, 4, 1, 0, bias=False)
        self.batch1=nn.BatchNorm2d(32 * 4)
        self.relu1=nn.ReLU(True)
        self.convt2=nn.ConvTranspose2d(32 *4, 32 * 2, 3, 2, 1, bias=False)
        self.batch2=nn.BatchNorm2d(32 * 2)
        self.relu2=nn.ReLU(True)
        self.convt3=nn.ConvTranspose2d(32 * 2, 32, 4, 2, 1, bias=False)
        self.batch3=nn.BatchNorm2d(32)
        self.relu3=nn.ReLU(True)
        self.convt4=nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)
        self.tanh=nn.Tanh()



    def forward(self, x):
        out=self.convt1(x)
        out=self.batch1(out)
        out=self.relu1(out)
        out=self.convt2(out)
        out=self.batch2(out)
        out=self.relu2(out)
        out=self.convt3(out)
        out=self.batch3(out)
        out=self.relu3(out)
        out=self.convt4(out)
        out=self.tanh(out)
        return out
