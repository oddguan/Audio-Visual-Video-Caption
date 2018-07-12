import torch
import torch.nn as nn 
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(Attention, self).__init__()
        self.linear1 = nn.Linear(dim*2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
    
    def forward(self, hidden_state, encoder_outputs):
        '''
            hidden_state.shape = 1, batch_size, dim_hidden
            encoder_output.shape = batch_size, length, dim_hidden
        '''
        batch_size, _len, _ = encoder_outputs.shape
        hidden_state = hidden_state.repeat(_len, 1, 1)
        hidden_state = torch.transpose(hidden_state, 0, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, _len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs)
        return context


