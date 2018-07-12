import json
import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from moviepy.video.io.VideoFileClip import VideoFileClip

class VideoAudioDataset(Dataset):
    def __init__(self, opt, mode):
        super(VideoAudioDataset, self).__init__()
        self.mode = mode

        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))
        self.feats_dir = opt['output_dir']
        self.video_dir = opt['video_dir']
        print('load feats from %s' % (self.feats_dir))
        self.max_len = opt['max_len']
        print('max sequence length in data is', self.max_len)

        self.max_video_duration = opt['max_video_duration']

    def __getitem__(self, ix):
        if self.mode == 'val':
            ix += len(self.splits['train'])
        elif self.mode == 'test':
            ix = ix + len(self.splits['train']) + len(self.splits['val'])

        image_feats = np.load(os.path.join(self.feats_dir+'video%i'%(ix), 'video.npy'))
        if os.path.exists(os.path.join(self.feats_dir+'video%i'%(ix), 'audio.npy')):
            audio_mfcc = np.load(os.path.join(self.feats_dir+'video%i'%(ix), 'audio.npy'))
            video_length = audio_mfcc.shape[0] / 32
            print(audio_mfcc.shape)
            audio_mfcc = np.pad(audio_mfcc, ((0, self.max_video_duration-round(video_length)), (0, 0)), 'constant')
            print(audio_mfcc.shape)
        else:
            duration = VideoFileClip(self.video_dir+'/'+'video%i'%(ix)+'.mp4').duration
            video_length = round(duration)
            audio_mfcc = np.zeros((32*self.max_video_duration, 20))
        self.video_length = video_length
        mask = np.zeros(self.max_len)
        label = np.zeros(self.max_len)
        captions = self.captions['video%i' % (ix)]['final_captions']
        gts = np.zeros((len(captions), self.max_len))
        for i, cap in enumerate(captions):
            if len(cap) > self.max_len:
                cap = cap[:self.max_len]
                cap[-1] = '<eos>'
            for j, w in enumerate(cap):
                gts[i, j] = self.word_to_ix[w]

        cap_ix = random.randint(0, len(captions)-1)
        label = gts[cap_ix]
        non_zero = (label == 0).nonzero()
        mask[:int(non_zero[0][0])+1]=1

        print(audio_mfcc.shape)
        data = dict()
        data['image_feats'] = torch.from_numpy(image_feats).type(torch.FloatTensor)
        data['audio_mfcc'] = torch.from_numpy(audio_mfcc).type(torch.FloatTensor)
        data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
        data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
        data['video_ids'] = 'video%i' % (ix)
        data['gts'] = torch.from_numpy(gts).long()
        return data

    def __len__(self):
        return len(self.splits[self.mode])
    
    def get_vocab_size(self):
        return len(self.ix_to_word)
    
    def get_vocab(self):
        return self.ix_to_word
    
    def get_video_length(self):
        return self.video_length
        