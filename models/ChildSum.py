import torch
import torch.nn as nn 
import torch.nn.functional as F

class ChildSum(nn.Module):
    def __init__(self, dim_vid=512, dim_aud=512):
        super(ChildSum, self).__init__()
        self.dim_vid = dim_vid
        self.dim_aud = dum_aud
        i1 = nn.Linear(dim_vid, dim_vid, bias=True)
        i2 = nn.Linear(dim_aud, dim_aud, bias=True)
        g1 = nn.Linear(dim_vid, dim_vid, bias=True)
        g2 = nn.Linear(dim_aud, dim_aud, bias=True)
        f1 = nn.Linear(dim_vid, dim_vid, bias=True)
        f2 = nn.Linear(dim_aud, dim_aud, bias=True)
        o1 = nn.Linear(dim_vid, dim_vid, bias=True)
        o2 = nn.Linear(dim_aud, dim_aud, bias=True)
    
    def forward(self, h1, h2, c1, c2):
        i = F.sigmoid(i1(h1)+i2(h2))
        g = F.tanh(g1(h1)+g2(h2))
        f = F.sigmoid(f1(h1)+f2(h2))
        o = F.sigmoid(o1(h1)+o2(h2))

        c = torch.bmm(i, g)+torch.bmm(f1, c1)+torch.bmm(f2, c2)
        h = torch.bmm(o, F.tanh(c))
        return (h, c)
