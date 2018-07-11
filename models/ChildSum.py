import torch
import torch.nn as nn 
import torch.nn.functional as F

class ChildSum(nn.Module):
    def __init__(self, dim_hidden):
        super(ChildSum, self).__init__()
        self.i1 = nn.Linear(dim_hidden, dim_hidden)
        self.i2 = nn.Linear(dim_hidden, dim_hidden)
        self.g1 = nn.Linear(dim_hidden, dim_hidden)
        self.g2 = nn.Linear(dim_hidden, dim_hidden)
        self.f1 = nn.Linear(dim_hidden, dim_hidden)
        self.f2 = nn.Linear(dim_hidden, dim_hidden)
        self.o1 = nn.Linear(dim_hidden, dim_hidden)
        self.o2 = nn.Linear(dim_hidden, dim_hidden)
    
    def forward(self, h1, h2, c1, c2):
        i = F.sigmoid(self.i1(h1)+self.i2(h2))
        g = F.tanh(self.g1(h1)+self.g2(h2))
        f_1 = F.sigmoid(self.f1(h1))
        f_2 = F.sigmoid(self.f2(h2))
        o = F.sigmoid(self.o1(h1)+self.o2(h2))
        c = i * g + f_1 * c1 + f_2 * c2
        h = o * F.tanh(c)
        return (h, c)
