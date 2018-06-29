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
        audio = output_dir + audio
        video_id = audio.split("/")[-1].split(".")[0]
        dst = output_dir + video_id
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.mkdir(dst)
        with open(os.devnull, 'w') as ffmpeg_log:
            command = 'ffmpeg -i ' + audio + ' -f segment -segment_time 1 -c copy ' + dst+ '/' + '%02d.wav'
            subprocess.call(command, shell=True, stdout=ffmpeg_log, stderr=ffmpeg_log)
        for segment in os.listdir(dst):
            segment = dst + '/' + segment
            sample_rate, audio_info = wavfile.read(segment)
            #print(audio_info.shape)
            audio_info = audio_info.astype(np.float32)
            mfcc_feats = mfcc(audio_info, sr=sample_rate)
            #print(mfcc_feats.shape)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, 
    help='The video dir that one would like to extract audio file from')
    parser.add_argument('--output_dir', type=str, 
    help='The audio file output directory')
    parser.add_argument('--output_channels', type=int, default=1, 
    help='The number of output audio channels, default to 1')
    parser.add_argument('--output_frequency', type=int, default=16000, 
    help='The output audio frequency in Hz, default to 16000')
    parser.add_argument('--band_width', type=int, default=160, 
    help='Bandwidth specified to sample the audio (unit in kbps), default to 160')
    
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
    print('done')

if __name__ == '__main__':
    main()