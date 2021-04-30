import torch.nn as nn



class LinearGenerator(nn.Module):
    def __init__(self, z_dim):
        super(LinearGenerator, self).__init__()
        self.z_dim= z_dim

        self.linear=nn.Linear(z_dim,256)
        self.relu=nn.LeakyReLU(0.2)
        self.linear2=nn.Linear(256, 512)
        self.relu2=nn.LeakyReLU(0.2)
        self.linear3=nn.Linear(512, 1024)
        self.relu3=nn.LeakyReLU(0.2)
        self.linear4=nn.Linear(1024,784)
        self.tan=nn.Tanh()



    def forward(self, x):
        out=self.linear(x)
        out=self.relu(out)
        out=self.linear2(out)
        out=self.relu2(out)
        out=self.linear3(out)
        out=self.relu3(out)
        out=self.linear4(out)
        out=self.tan(out)

        out=out.view(x.size(0),1,28,28)
        return out