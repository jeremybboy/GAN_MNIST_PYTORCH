import torch.nn as nn

class LinearDiscriminator(nn.Module):
    def __init__(self):
        super(LinearDiscriminator, self).__init__()

        self.linear=nn.Linear(784,1024)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.3)
        self.linear2=nn.Linear(1024,512)
        self.relu2=nn.ReLU()
        self.dropout2=nn.Dropout(0.3)
        self.linear3=nn.Linear(512,256)
        self.relu3=nn.ReLU()
        self.dropout3=nn.Dropout(0.3)
        self.linear4=nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, x):
        x = x.view(-1, 784)
        out=self.linear(x)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.linear2(out)
        out=self.relu2(out)
        out=self.dropout2(out)
        out=self.linear3(out)
        out=self.relu3(out)
        out=self.dropout3(out)
        out=self.linear4(out)
        out=self.sigmoid(out)
        #using BCE with logits
        return out

