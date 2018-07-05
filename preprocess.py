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

C, H, W = 3, 224, 224

def extract_frame(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            command = 'ffmpeg -i ' + video + ' -vf fps=15 ' + '{0}/%06d.jpg'.format(dst)
            subprocess.call(command, stdout=ffmpeg_log, stderr=ffmpeg_log)

def extract_image_feats(opt, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = opt['output_dir']
    
    print('extracting video features...')
    print('save video features to %s' % (dir_fc))

    video_list = glob.glob(os.path.join(opt['video_dir'], '*.mp4'))
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = dir_fc +video_id
        extract_frame(video, dst)

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))
        images = torch.zeros((len(image_list), C, H, W))
        for i in range(len(image_list)):
            img = load_image_fn(image_list[i])
            images[i] = img
        
        with torch.no_grad():
            image_feats = model(images.cuda().squeeze())
        image_feats = image_feats.cpu().numpy()
        outfile = os.path.join(dst, 'video.npy')
        np.save(outfile, image_feats)
        for file in os.listdir(dst):
            if file.endswith('.jpg'):
                os.remove(os.path.join(dst, file))



def vToA(opt):
    video_dir = opt['video_dir']
    dst = opt['output_dir']
    
    band_width = opt['band_width']
    output_channels = opt['output_channels']
    output_frequency = opt['output_frequency']
    # print(video_id)
    if os.path.exists(dst):
        print(" cleanup: " + dst + "/")
        shutil.rmtree(dst)
    os.makedirs(dst)
    for video in tqdm(os.listdir(video_dir)):
        video = video_dir + '/' + video
        video_id = video.split("/")[-1].split(".")[0]
        with open(os.devnull, "w") as ffmpeg_log:
            
            command = 'ffmpeg -i ' + video + ' -ab ' + str(band_width) + 'k -ac ' + str(output_channels) + ' -ar ' + str(output_frequency) + ' -vn ' + dst + '/' + video_id + '.wav'
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)



def split_audio(opt):
    print('splitting audios...')
    output_dir = opt['output_dir']
    print('output directory: '+output_dir)
    for audio in os.listdir(output_dir):
        audio = output_dir + '/' + audio
        video_id = audio.split("/")[-1].split(".")[0]
        dst = output_dir + '/' + video_id
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.mkdir(dst)
        with open(os.devnull, 'w') as ffmpeg_log:
            command = 'ffmpeg -i ' + audio + ' -f segment -segment_time 1 -c copy ' + dst+ '/' + '%02d.wav'
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
        
        output = np.zeros((20, 0))
        for segment in os.listdir(dst):
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
        print(output.shape)
        outfile = os.path.join(dst, 'audio.npy')
        np.save(outfile, output)
        for file in os.listdir(dst):
            if file.endswith('.wav'):
                os.remove(os.path.join(dst, file))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, 
    help='The video dir that one would like to extract audio file from')
    parser.add_argument('--output_dir', type=str, 
    help='The file output directory')
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
    opt = parser.parse_args()
    opt=vars(opt)

    vToA(opt)
    split_audio(opt)
    print('cleaning up original .wav files...')
    dir = opt['output_dir']
    dir = os.listdir(dir)
    for file in dir:
        if file.endswith('.wav'):
            os.remove(os.path.join(opt['output_dir'], file))
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu']
    # if opt['model'] == 'resnet152':
    #     C, H, W = 3, 224, 224
    #     model = pretrainedmodels.resnet152(pretrained='imagenet')
    #     load_image_fn = utils.LoadTransformImage(model)
    # elif opt['model'] == 'inception_v3':
    #     C, H, W = 3, 299, 299
    #     model = pretrainedmodels.inceptionv3(pretrained='imagenet')
    #     load_image_fn = utils.LoadTransformImage(model)
    # elif opt['model'] == 'vgg16':
    #     C, H, W = 3, 224, 224
    #     model = pretrainedmodels.vgg16(pretrained=True)
    #     load_image_fn = utils.LoadTransformImage(model)
    # else:
    #     print('The image model is not supported')
    
    # model.last_linear = utils.Identity()
    # model = nn.DataParallel(model)

    # model = model.cuda()
    # extract_image_feats(opt, model, load_image_fn)

if __name__ == '__main__':
    main()