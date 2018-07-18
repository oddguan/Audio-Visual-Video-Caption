import json
import os
import subprocess
import glob
import shutil
from pretrainedmodels import resnet152
from librosa.feature import mfcc
from scipy.io import wavfile
import numpy as np
import pretrainedmodels.utils as utils

import torch
import torch.nn as nn 
from models import MultimodalAtt
import NLUtils


def vToA(path):
    print('converting video to audio...')
    band_width = 160
    output_channels = 1
    output_frequency = 16000
    video_name = path.split("/")[-1].split(".")[0]
    dst = os.path.join(path.split('/')[0], 'info')
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)
    with open(os.devnull, "w") as ffmpeg_log:
        command = 'ffmpeg -i ' + path + ' -ab ' + str(band_width) + 'k -ac ' + str(output_channels) + ' -ar ' + str(output_frequency) + ' -vn ' + dst + '/' + video_name + '.wav'
        subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
    return os.path.join(dst, video_name+'.wav')


def split_audio(wav_path):
    print('splitting audios...')
    dst = os.path.join(wav_path.split('/')[0], 'info')
    with open(os.devnull, 'w') as ffmpeg_log:
        command = 'ffmpeg -i ' + wav_path + ' -f segment -segment_time 1 -c copy ' + os.path.join(dst,'%02d.wav')
        subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
    os.remove(wav_path)
    output = np.zeros((20, 0))
    for segment in os.listdir(dst):
        segment = os.path.join(dst, segment)
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
    
    for file in os.listdir(dst):
        if file.endswith('.wav'):
            os.remove(os.path.join(dst, file))

    return output.T


def extract_image_feats(video_path):
    print('extracting image features...')
    model = resnet152(pretrained='imagenet')
    model = model.cuda()
    model = nn.DataParallel(model)
    model.last_linear = utils.Identity()
    model.eval()
    C, H, W = 3, 224, 224
    load_image_fn = utils.LoadTransformImage(model)
    dst = os.path.join(video_path.split('/')[0], 'info')
    with open(os.devnull, "w") as ffmpeg_log:
        command = 'ffmpeg -i ' + video_path + ' -vf scale=400:300 ' + '-qscale:v 2 '+ '{0}/%06d.jpg'.format(dst)
        subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
    image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
    samples = np.round(np.linspace(0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = torch.zeros((len(image_list), C, H, W))
    for i in range(len(image_list)):
        img = load_image_fn(image_list[i])
        images[i] = img
    with torch.no_grad():
        image_feats = model(images.cuda().squeeze())
    image_feats = image_feats.cpu().numpy()
    for file in os.listdir(dst):
        if file.endswith('.jpg'):
            os.remove(os.path.join(dst, file))

    return image_feats



def main():
    # video_path = input('en In the Shell you should seter the path to the video:')
    video_path = '../video4.mp4'
    # model_path = input('enter the model path: ')
    model_path = 'save/vanilla/model_290.pth'
    wav_path = vToA(video_path)
    audio_mfcc = split_audio(wav_path)
    audio_mfcc = torch.from_numpy(audio_mfcc).type(torch.FloatTensor).unsqueeze(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    image_feats = extract_image_feats(video_path)
    image_feats = torch.from_numpy(image_feats).type(torch.FloatTensor).unsqueeze(0)
    print('generating caption...')
    model = MultimodalAtt(16860, 28, 1024, 512, rnn_dropout_p=0)
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    opt = dict()
    opt['child_sum'] = True
    opt['temporal_attention'] = True
    opt['multimodel_attention'] = True
    with torch.no_grad():
        _, seq_preds = model(image_feats, audio_mfcc, mode='inference', opt=opt)
    vocab = json.load(open('data/info.json'))['ix_to_word']
    sent = NLUtils.decode_sequence(vocab, seq_preds)
    print(sent)

if __name__ == '__main__':
    main()
