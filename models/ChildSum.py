import torch
import torch.nn as nn 
import torch.nn.functional as F

class ChildSum(nn.Module):
    def __init__(self, dim_vid=512, dim_aud=512):
        super(ChildSum, self).__init__()
        self.dim_vid = dim_vid
        self.dim_aud = dim_aud
        self.i1 = nn.Linear(dim_vid, dim_vid, bias=True)
        self.i2 = nn.Linear(dim_aud, dim_aud, bias=True)
        self.g1 = nn.Linear(dim_vid, dim_vid, bias=True)
        self.g2 = nn.Linear(dim_aud, dim_aud, bias=True)
        self.f1 = nn.Linear(dim_vid, dim_vid, bias=True)
        self.f2 = nn.Linear(dim_aud, dim_aud, bias=True)
        self.o1 = nn.Linear(dim_vid, dim_vid, bias=True)
        self.o2 = nn.Linear(dim_aud, dim_aud, bias=True)
    
    def forward(self, h1, h2, c1, c2):
        i = F.sigmoid(self.i1(h1)+self.i2(h2))
        g = F.tanh(self.g1(h1)+self.g2(h2))
        f_1 = self.f1(h1)
        f_2 = self.f2(h2)
        f = F.sigmoid(f_1, f_2)
        o = F.sigmoid(self.o1(h1)+self.o2(h2))

        c = torch.bmm(i, g)+torch.bmm(f_1, c1)+torch.bmm(f_2, c2)
        h = torch.bmm(o, F.tanh(c))
        return (h, c)
