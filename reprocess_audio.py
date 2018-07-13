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

import subprocess
import argparse
from librosa.feature import mfcc

import librosa
import os
import shutil
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import glob

import torch
import torch.nn as nn

import pretrainedmodels
import pretrainedmodels.utils as utils

def vToA(opt):
    video_dir = opt['video_dir']
    dst = opt['output_dir']
    
    band_width = opt['band_width']
    output_channels = opt['output_channels']
    output_frequency = opt['output_frequency']
    # print(video_id)
    # if os.path.exists(dst):
    #     print(" cleanup: " + dst + "/")
    #     shutil.rmtree(dst)
    # os.makedirs(dst)
    for video in tqdm(os.listdir(video_dir)):
        video = video_dir + '/' + video
        video_id = video.split("/")[-1].split(".")[0]
        with open(os.devnull, "w") as ffmpeg_log:
            
            command = 'ffmpeg -i ' + video + ' -ab ' + str(band_width) + 'k -ac ' + str(output_channels) + ' -ar ' + str(output_frequency) + ' -vn ' + dst + '/' + video_id + '.wav'
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)

def split_audio(opt):
    print('splitting audios...')
    npy_dir = opt['npy_dir']
    output_dir = opt['output_dir']
    print('output directory: '+npy_dir)
    for audio in tqdm(os.listdir(output_dir)):
        audio = os.path.join(output_dir, audio)
        video_id = audio.split("/")[-1].split(".")[0]
        dst = os.path.join(npy_dir, video_id)
        # if os.path.exists(dst):
        #     shutil.rmtree(dst)
        # os.mkdir(dst)
        with open(os.devnull, 'w') as ffmpeg_log:
            command = 'ffmpeg -i ' + audio + ' -f segment -segment_time 1 -c copy ' + dst+ '/' + '%02d.wav'
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
        
        output = np.zeros((20, 0))
        for segment in os.listdir(dst):
            if segment == 'audio.npy' or segment == 'video.npy':
                continue
            segment = dst + '/' + segment
            sample_rate, audio_info = wavfile.read(segment)
            audio_length = audio_info.shape[0]
            if audio_length<=16000:
                audio_info = np.pad(audio_info, (0, 16000-audio_length), 'constant', constant_values=0)
            else:
                audio_info = audio_info[0:16000]
            audio_info = audio_info.astype(np.float32)
            mfcc_feats = mfcc(audio_info, sr=sample_rate)
            #print(mfcc_feats.shape)
            output = np.concatenate((output, mfcc_feats), axis=1)
        #print(output.shape)
        video_length = output.shape[1] / 32
        output = np.pad(output, ((0,0),(0, 32*(opt['max_video_duration']-round(video_length)))), 'constant')
        outfile = os.path.join(dst, 'audio.npy')
        np.save(outfile, output.T)
        for file in os.listdir(dst):
            if file.endswith('.wav'):
                os.remove(os.path.join(dst, file))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, 
    help='The video dir that one would like to extract audio file from')
    parser.add_argument('--output_dir', type=str, 
    help='The audio file output directory')
    parser.add_argument('--npy_dir', type=str,
    help='the numpy array saving directory')
    parser.add_argument('--output_channels', type=int, default=1, 
    help='The number of output audio channels, default to 1')
    parser.add_argument('--output_frequency', type=int, default=16000, 
    help='The output audio frequency in Hz, default to 16000')
    parser.add_argument('--band_width', type=int, default=160, 
    help='Bandwidth specified to sample the audio (unit in kbps), default to 160')
    parser.add_argument('--model', type=str, default='resnet152', 
    help='The pretrained model to use for extracting image features, default to resnet152')
    parser.add_argument('--gpu', type=str, default='0', 
    help='The CUDA_VISIBLE_DEVICES argument, default to 0')
    parser.add_argument('--n_frame_steps', type=int, default=80,
    help='The number of frames to extract from a single video')
    parser.add_argument('--max_video_duration', type=int, default=35)
    opt = parser.parse_args()
    opt=vars(opt)

    #vToA(opt)
    split_audio(opt)

if __name__ == '__main__':
    main()
