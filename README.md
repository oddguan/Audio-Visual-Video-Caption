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

`$python train.py --video_dir video/directory --output_dir features/directory --gpu 0`

## Currently...

### 07/16/2018
I made sure that the training part is working without the childsum unit and multilevel attentions. The model is being trained and producing a decent result, and by decent I mean the training loss is lower than what the vanilla non-audio feature model has produced. 

By using 4 Nvidia GeForce 1080ti, one is used for the basic mean-pool model, and the other three are used for the mean-pool with multi-level attentions. For the latter model, the batch sized was down to 32 from 128, since the attention model has taken so much memories from GPU. Even using three GPUs instead of one, the model still can't fit into the memory. Hopefully the result won't be affected a lot by this.


## TODO
- Create a score benchmark for all of my implementations.
- To make sure the validation and the evaluation part works properly, after I have a spare gpu to test.
- If time permitted, I will come up with a CNN architecture to extract the audio feature and compare the result, or use the vggish model instead of a lstm extracting mfccs. 
- Utilize the dataset from Marc and Justin.


