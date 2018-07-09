import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .Attentions import NaiveAttention

class MultimodalAtt(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, dim_audio=32, 
    sos_id=1, eos_id=0, n_layers=1, rnn_cell='lstm', rnn_dropout_p=0.2, decoder_input_shape=20):
        super(MultimodalAtt, self).__init__()

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        if rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        
        self.dim_word = dim_word
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.max_len = max_len
        self.n_layers = n_layers
        self.dim_vid = dim_vid
        self.dim_audio = dim_audio
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.decoder_input_shape = 20

        self.video_rnn_encoder = self.rnn_cell(self.dim_vid, self.dim_hidden, self.n_layers, dropout=rnn_dropout_p, batch_first=True)
        self.audio_rnn_encoder = self.rnn_cell(self.dim_audio, self.dim_hidden, self.n_layers, dropout=rnn_dropout_p, batch_first=True)

        self.naive_fusion = nn.Linear(self.dim_hidden*2, dim_hidden, bias=False)
        self.fuse_input = nn.Linear(2, 1)
        self.decoder = self.rnn_cell(self.dim_hidden+self.dim_word, self.dim_hidden, n_layers, dropout=rnn_dropout_p, batch_first=True)

        self.embedding = nn.Embedding(self.dim_output, self.dim_word)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)




    def forward(self, image_feats, audio_feats, target_variable=None, mode='train', opt={}):
        batch_size, n_frames, _ = image_feats.shape
        _, n_mfcc, __ = audio_feats.shape
        padding_words = torch.zeros((batch_size, n_frames if n_frames>n_mfcc else n_mfcc, self.dim_word)).cuda()
        padding_frames = torch.zeros((batch_size, 1, self.dim_vid)).cuda()
        video_encoder_output, (video_hidden_state, video_cell_state) = self.video_rnn_encoder(image_feats)
        audio_encoder_output, (audio_hidden_state, audio_cell_state) = self.audio_rnn_encoder(audio_feats)
        decoder_h0 = torch.cat((video_hidden_state, audio_hidden_state), dim=2)
        decoder_h0 = F.tanh(self.naive_fusion(decoder_h0))
        decoder_c0 = video_cell_state + audio_cell_state

        decoder_input = pad_sequence([audio_encoder_output.squeeze(), video_encoder_output.squeeze()])
        decoder_input = torch.transpose(decoder_input, 1, 2)
        decoder_input = self.fuse_input(decoder_input)
        decoder_input = decoder_input.squeeze().unsqueeze(0)
        decoder_input = torch.cat((decoder_input, padding_words), dim=2)

        decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_h0, decoder_c0))
        seq_probs = list()
        seq_preds = list()
        if mode == 'train':
            for i in range(self.max_length - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.video_rnn_encoder.flatten_parameters()
                self.audio_rnn_encoder.flatten_parameters()
                self.decoder.flatten_parameters()
                output1, (decoder_h0, decoder_c0) = self.rnn1(padding_frames, (decoder_h0, decoder_c0))
                input2 = torch.cat((output1, current_words.unsqueeze(1)), dim=2)
                output2, (decoder_hidden, decoder_cell) = self.rnn2(input2, (decoder_hidden, decoder_cell))
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        # else:
        #     current_words = self.embedding(
        #         Variable(torch.LongTensor([self.sos_id] * batch_size)).cuda())
        #     for i in range(self.max_length - 1):
        #         self.rnn1.flatten_parameters()
        #         self.rnn2.flatten_parameters()
        #         output1, state1 = self.rnn1(padding_frames, state1)
        #         input2 = torch.cat(
        #             (output1, current_words.unsqueeze(1)), dim=2)
        #         output2, state2 = self.rnn2(input2, state2)
        #         logits = self.out(output2.squeeze(1))
        #         logits = F.log_softmax(logits, dim=1)
        #         seq_probs.append(logits.unsqueeze(1))
        #         _, preds = torch.max(logits, 1)
        #         current_words = self.embedding(preds)
        #         seq_preds.append(preds.unsqueeze(1))
        #     seq_probs = torch.cat(seq_probs, 1)
        #     seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds

        


