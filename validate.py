import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

import os
import argparse
import json
import glob
from tqdm import tqdm
from pandas.io.json import json_normalize

import NLUtils
from cocoeval import suppress_stdout_stderr, COCOScorer
from dataloader import VideoAudioDataset
from models import MultimodalAtt

def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    return gts


def eval(model, crit, dataset, vocab, opt, model_path):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    samples = {}
    for data in loader:
        # forward the model to get loss
        image_feats = data['image_feats'].cuda()
        audio_mfcc = data['audio_mfcc'].cuda()
        video_ids = data['video_ids']
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(image_feats, audio_mfcc, mode='inference', opt=opt)

        sents = NLUtils.decode_sequence(vocab, seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    validation_file_name = opt['model_directory'].split('/')[-1]+'_val_score.txt'
    with open(os.path.join(opt["results_path"], validation_file_name), 'a') as scores_table:
        scores_table.write(model_path.split('/')[-1]+': '+json.dumps(results[0]) + "\n")

def main(opt):
    dataset = VideoAudioDataset(opt, 'val')
    opt['vocab_size'] = dataset.get_vocab_size()
    model = MultimodalAtt(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], opt['dim_word'], dim_vid=opt['dim_vid'],
    n_layers=opt['num_layers'], rnn_cell=opt['rnn_type'], rnn_dropout_p=opt['rnn_dropout_p']).cuda()
    model = nn.DataParallel(model)
    crit = NLUtils.LanguageModelCriterion()
    for model_path in tqdm(glob.glob(os.path.join(opt['model_directory'],'*.pth'))):
        model.load_state_dict(torch.load(model_path))
        eval(model, crit, dataset, dataset.get_vocab(), opt, model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--model_directory', type=str, required=True,
                        help='path to saved model directory')
    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    # parser.add_argument('--sample_max', type=int, default=1,
    #                     help='0/1. whether sample max probs  to get next word in inference stage')
    # parser.add_argument('--temperature', type=float, default=1.0)
    # parser.add_argument('--beam_size', type=int, default=1,
    #                     help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)