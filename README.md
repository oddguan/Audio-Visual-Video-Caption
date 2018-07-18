# Audio Visual Video Captioning

This is the project that I built for video captioning with the MSR-VTT dataset by using the pytorch framework, which involves both visual and audio information.

Video's visual content are preprocessed into a fixed number of frames, feed into a pretrained deep CNN (ResNet 152 for example) to extract features, and feed into a LSTM encoder. For the audio content, They are preprocessed into MFCCs and feed into another LSTM encoder. The outputs and hidden states of both LSTM encoders are then combined by average pooling (or multi-model attentions, and childsum unit<sup>[1](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/coi110-xuA.pdf)</sup>) and further feed into the LSTM decoder for generating the captions.

The basic structure of the whole project are imported from [here](https://github.com/xiadingZ/video-caption.pytorch). 

To run the project, you need the following dependencies:

- python 3
- [pytorch 0.4.0](https://pytorch.org/)
- [cuda 9.1](https://developer.nvidia.com/cuda-91-download-archive)
- [ffmpeg](https://www.ffmpeg.org/)
- [tqdm](https://tqdm.github.io/)
- [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)
- [librosa](https://github.com/librosa/librosa)

## Steps To Run the Model

the first step would be preprocess the video and captions

`$ python preprocess.py --video_dir path/to/the/training/video/directory --output_dir path/to/the/features/output/dir`

This is will preprocess video contents into extracted features and audio mfcc in `.npy` fashion.

`$ python preprocess_vocab.py`

This will generate the vocab base json file.

To train the model, run `train.py`, and all of the option inputs are in the `opts.py` file for you to explore. One example to run the training process would be:

`$python train.py --video_dir video/directory --output_dir features/directory --gpu 0`

## Results comparison

The following table shows some state-of-the-art results of video captioning models on MSR-VTT, and my own results on the problem.

| Methods                 | BLEU@4 | METEOR | CIDEr-D | ROUGE-L|
|:------------------------|:------:|:------:|:-----:|:------:|
|MA-LSTM (G+C+A)(childsum)<sup>[1]</sup>|36.5|26.5 |41.0|59.8|
|hLSTMat (R)<sup>[2]</sup>|38.3|26.3|-|-|
|R+MCNN+MCF-matrix multiply <sup>[3]</sup>|38.1|27.2|42.1|-|
|**My own methods**           |  |  |  |
|Vanilla S2VT (without audio)<sup>[4]</sup>|0.29|0.25|-|0.55|
|Vanilla mean-pool|33.7|26.6|41.4|57.9|
|Multimodal Attention|35.8|26.3|40.2|57.9|

## Currently...

### 07/18/2018
Finished evaluating two methods: the vanilla mean-pool and the multimodal attention. Still left with the child-sum unit to go. Two methods have similar performance, while the model with attention requires more GPU memory to both train and evaluate. They still can't beat the state-of-the-art results, but they perform fairly well, and it is pretty clear that the audio feature has helped a lot for the model to better understand the video content, compare to my own implementation of the vanilla S2VT model presented by [this](https://arxiv.org/pdf/1505.00487.pdf) paper. 

### 07/16/2018
I made sure that the training part is working without the childsum unit and multilevel attentions. The model is being trained and producing a decent result, and by decent I mean the training loss is lower than what the vanilla non-audio feature model has produced. 

By using 4 Nvidia GeForce 1080ti, one is used for the basic mean-pool model, and the other three are used for the mean-pool with multi-level attentions. For the latter model, the batch size was down to 32 from 128, since the attention model has taken so much memories from GPU. Even using three GPUs instead of one, the model still can't fit into the memory. Hopefully the result won't be affected a lot by this.


## TODO
- Create a score benchmark for all of my implementations (still ongoing for more implementations).
- If time permitted, I will come up with a CNN architecture to extract the audio feature and compare the result, or use the vggish model instead of a lstm extracting MFCCs. 
- Utilize the dataset from Marc and Justin.
- Comment the whole project.


## References

<sup>[1]</sup> Xu, Jun, et al. "Learning Multimodal Attention LSTM Networks for Video Captioning." Proceedings of the 2017 ACM on Multimedia Conference. ACM, 2017.

<sup>[2]</sup> Song, Jingkuan, et al. "Hierarchical LSTM with adjusted temporal attention for video captioning." arXiv preprint arXiv:1706.01231 (2017).

<sup>[3]</sup> Wu, Aming, and Yahong Han. "Multi-modal Circulant Fusion for Video-to-Language and Backward." IJCAI. 2018.

<sup>[4]</sup> Venugopalan, Subhashini, et al. "Sequence to sequence-video to text." Proceedings of the IEEE international conference on computer vision. 2015.