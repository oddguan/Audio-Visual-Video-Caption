import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from .Attention import Attention
from .ChildSum import ChildSum

class MultimodalAtt(nn.Module):

    def __init__(self, vocab_size, max_len, dim_hidden, dim_word, dim_vid=2048, 
    dim_audio=20, sos_id=1, eos_id=0, n_layers=1, rnn_dropout_p=0):
        super(MultimodalAtt, self).__init__()

        self.rnn_cell = nn.LSTM
        
        # dimension of the word embedding layer
        self.dim_word = dim_word
        # Output size would be a one-dimensional vector of the vocab base
        self.dim_output = vocab_size
        # The hidden size of the LSTM cells output
        self.dim_hidden = dim_hidden
        # Define the max length of either the generated sentence or training caption
        self.max_len = max_len
        # Number of layers for the LSTM cells. Default to 1
        self.n_layers = n_layers
        # Dimension of the video feature. Default to 2048
        self.dim_vid = dim_vid
        # Dimension of the audio feature. Default to 20 (extract 20 MFCCs for each time step)
        self.dim_audio = dim_audio
        # The ix in the vocab base for the <SOS> signal
        self.sos_id = sos_id
        # Same as above for <EOS>
        self.eos_id = eos_id

        # Define LSTM encoders
        self.video_rnn_encoder = self.rnn_cell(self.dim_vid, self.dim_hidden, 
        self.n_layers, dropout=rnn_dropout_p, batch_first=True)
        self.audio_rnn_encoder = self.rnn_cell(self.dim_audio, self.dim_hidden, 
        self.n_layers, dropout=rnn_dropout_p, batch_first=True)

        # Define Attention layers and child-sum unit
        self.TemporalAttention_vid = Attention(dim_hidden)
        self.TemporalAttention_aud = Attention(dim_hidden)
        self.MultiModelAttention = Attention(dim_hidden)
        self.ChildSum = ChildSum(dim_hidden)

        #Define the LSTM decoder to generate caption
        self.decoder = self.rnn_cell(self.dim_word+self.dim_hidden, self.dim_hidden, 
        n_layers, dropout=rnn_dropout_p, batch_first=True)

        # Embedding layer of the vocab one-hot vector
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)
        self.out = nn.Linear(self.dim_hidden, self.dim_output)


    def forward(self, image_feats, audio_feats, target_variable=None, mode='train', opt={}):
        # load features by dataloader
        # shape = (batch_size, number of frames, dimension of the video feature)
        batch_size, n_frames, _ = image_feats.shape
        # shape = (batch_size, number of time steps, dimension of the audio feature)
        _, __ , n_mfcc = audio_feats.shape
        # Encode the inputs
        video_encoder_output, (video_hidden_state, video_cell_state) = self.video_rnn_encoder(image_feats)
        audio_encoder_output, (audio_hidden_state, audio_cell_state) = self.audio_rnn_encoder(audio_feats)
        if opt['child_sum']:
            decoder_hidden, decoder_cell = self.ChildSum(video_hidden_state, audio_hidden_state, 
            video_cell_state, audio_cell_state)
        else:
            decoder_hidden = video_hidden_state + audio_hidden_state
            decoder_cell = video_cell_state + audio_cell_state
        
        if opt['temporal_attention']:
            vid_context = self.TemporalAttention_vid(decoder_hidden, video_encoder_output)
            aud_context = self.TemporalAttention_aud(decoder_hidden, audio_encoder_output)    
        else:
            vid_context = video_encoder_output.mean(1).unsqueeze(1)
            aud_context = audio_encoder_output.mean(1).unsqueeze(1)
        
        context = torch.cat((vid_context, aud_context), dim=1)
        if opt['multimodel_attention']:
            decoder_input = self.MultiModelAttention(decoder_hidden, context)
        else:
            decoder_input = context.mean(1).unsqueeze(1)

        seq_probs = list()
        seq_preds = list()
        if mode == 'train':
            for i in range(self.max_len - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.video_rnn_encoder.flatten_parameters()
                self.audio_rnn_encoder.flatten_parameters()
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                
                if opt['temporal_attention']:
                    vid_context = self.TemporalAttention_vid(decoder_hidden, video_encoder_output)
                    aud_context = self.TemporalAttention_aud(decoder_hidden, audio_encoder_output)    
                else:
                    vid_context = video_encoder_output.mean(1).unsqueeze(1)
                    aud_context = audio_encoder_output.mean(1).unsqueeze(1)
        
                context = torch.cat((vid_context, aud_context), dim=1)
                if opt['multimodel_attention']:
                    decoder_input = self.MultiModelAttention(decoder_hidden, context)
                else:
                    decoder_input = context.mean(1).unsqueeze(1)
                
                output = self.out(decoder_output)
                seq_probs.append(output)
            seq_probs = torch.cat(seq_probs, 1)

        elif mode == 'inference':
            current_words = self.embedding(torch.cuda.LongTensor([self.sos_id] * batch_size))

            for i in range(self.max_len-1):
                self.video_rnn_encoder.flatten_parameters()
                self.audio_rnn_encoder.flatten_parameters()
                self.decoder.flatten_parameters()
                decoder_input = torch.cat((decoder_input, current_words.unsqueeze(1)), dim=2)
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
                
                if opt['temporal_attention']:
                    vid_context = self.TemporalAttention_vid(decoder_hidden, video_encoder_output)
                    aud_context = self.TemporalAttention_aud(decoder_hidden, audio_encoder_output)    
                else:
                    vid_context = video_encoder_output.mean(1).unsqueeze(1)
                    aud_context = audio_encoder_output.mean(1).unsqueeze(1)
        
                context = torch.cat((vid_context, aud_context), dim=1)
                if opt['multimodel_attention']:
                    decoder_input = self.MultiModelAttention(decoder_hidden, context)
                else:
                    decoder_input = context.mean(1).unsqueeze(1)
                logits = F.log_softmax(self.out(decoder_output).squeeze(1), dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds

        


