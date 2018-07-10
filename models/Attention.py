import torch
import torch.nn as nn 
import torch.nn.functional as F

# This is the model that multiply a beta factor to each
# feature vector from each modality and then feed into the decoder RNN
class NaiveAttention(nn.Module):
    def __init__(self, dim):
        super(NaiveAttention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim*2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
    
    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
    
    def forward(self, audio_hidden_state, audio_outputs, video_hidden_state, video_outputs):
        return


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super(TemporalAttention, self).__init__()
        self.linear1 = nn.Linear(dim*2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
    
    def forward(self, hidden_state, encoder_output):
        return

