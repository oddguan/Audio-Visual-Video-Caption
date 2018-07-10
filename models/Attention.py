import torch
import torch.nn as nn 
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(dim*2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
    
    def forward(self, hidden_state, encoder_outputs):
        batch_size, _len, _ = encoder_outputs.shape
        hidden_state = hidden_state.unsqueeze(1).repeat(1, _len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, _len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context


