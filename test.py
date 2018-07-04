# import numpy as np 
# import torch

# # f = np.zeros((20, 0))
# # a = np.zeros((20,95))
# # b = np.zeros((20, 97))
# # c = np.zeros((20, 97))
# # d = np.zeros((20, 97))
# # e = np.zeros((20, 97))
# # f = np.concatenate((f,a), axis=1)
# # print(f.shape)

# a = torch.zeros((200, 10, 10))
# b = a[0:15]
# print(b.shape)

# import torch
# import torch.nn as nn 
# class MultimodalAtt(nn.Module):

#     def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, dim_audio=32, 
#     sos_id=1, eos_id=0, n_layers=1, rnn_cell='lstm', rnn_dropout_p=0.2):
#         super(MultimodalAtt, self).__init__()

#         if rnn_cell.lower() == 'lstm':
#             self.rnn_cell = nn.LSTM
#         if rnn_cell.lower() == 'gru':
#             self.rnn_cell = nn.GRU
        
#         self.dim_word = dim_word
#         self.dim_output = vocab_size
#         self.dim_hidden = dim_hidden
#         self.max_len = max_len
#         self.n_layers = n_layers
#         self.dim_vid = dim_vid
#         self.dim_audio = dim_audio
#         self.sos_id = sos_id
#         self.eos_id = eos_id

#         self.video_rnn_encoder = self.rnn_cell(self.dim_vid, self.dim_hidden, self.n_layers, dropout=rnn_dropout_p)
#         self.audio_rnn_encoder = self.rnn_cell(self.dim_audio, self.dim_hidden, self.n_layers, dropout=rnn_dropout_p)

#         self.embedding = nn.Embedding(self.dim_output, self.dim_word)
#         self.out = nn.Linear(self.dim_hidden, self.dim_output)


#     def forward(self, image_feats, audio_feats, target_variable=None, mode='train', opt={}):
#         n_frames, batch_size, _ = image_feats.shape
#         # padding_words = torch.zeros((batch_size, n_frames, self.dim_word))
#         # padding_frames = torch.zeros((batch_size, 1, self.dim_vid))
#         video_encoder_output, (video_hidden_state, video_cell_state) = self.video_rnn_encoder(image_feats)
#         audio_encoder_output, (audio_hidden_state, audio_cell_state) = self.audio_rnn_encoder(audio_feats)
#         return video_encoder_output, video_hidden_state, video_cell_state, audio_encoder_output, audio_hidden_state, audio_cell_state

# def main():
#     model = MultimodalAtt(18600, 28, 512, 512, rnn_dropout_p=0)
#     model = model.cuda()
#     model.eval()
#     image_feats = torch.zeros((15, 1, 2048)).cuda()
#     audio_feats = torch.zeros((20, 1, 32)).cuda()

#     veo, vhs, vcs, aeo, ahs, acs = model(image_feats, audio_feats)
#     print(veo.shape)
#     print(vhs.shape)
#     print(vcs.shape)
#     print(aeo.shape)
#     print(ahs.shape)
#     print(acs.shape)

# if __name__ == '__main__':
#     main()

import torch
import torch.nn as nn
a = torch.zeros((30, 50))
linear = nn.Linear(50, 40)
print(linear(a).shape)