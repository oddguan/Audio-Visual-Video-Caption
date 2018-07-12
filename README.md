# video_caption_v2

This is the project that I built for video captioning with the MSR-VTT dataset by using the pytorch framework, which involves both video and audio information.

Videos visual content are preprocessed into a fixed number of frames, feed into a pretrained deep CNN (ResNet 152 for example) to extract features, and feed into a LSTM encoder. For the audio content, They are preprocessed into mfccs and feed into another LSTM encoder. The outputs and hidden states of both LSTM encoders are then combined by average pooling (or multi-model attention attentions) and further feed into the LSTM decoder for generating the captions.

To run the project, you need the following dependencies:

- python 3
- pytorch 0.4.0
- cuda
- ffmpeg
- tqdm
- pretrainedmodels 

## Steps To Run the Model

the first step would be preprocess the video and captions

`$ python preprocess.py --video_dir path/to/the/training/video/directory --output_dir path/to/the/features/output/dir`

This is will preprocess video contents into extracted features and audio mfcc in `.npy` fasion.

`$ python preprocess_vocab.py`

This will generate the vocab base json file.

To train the model, run `train.py`, and all of the option inputs are in the `opts.py` file for you to explore. One example to run the training process would be:

`$python train.py --video_dir video/directory --output_dir features/directory --gpu 0 --rnn_dropout_p 0`


## TODO

`eval.py`

Attention models don't work since it runs out of memory everytime. waiting for the results of non-attended model.

