import torch.nn as nn
from dataloader import VideoAudioDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
import os
import json 
from models import MultimodalAtt
from NLUtils import LanguageModelCriterion
import opts

def train(loader, model, crit, optimizer, lr_scheduler, opt):
    model.train()
    model = nn.DataParallel(model)
    for epoch in range(opt['epochs']):
        lr_scheduler.step()
        iteration = 0

        for data in loader:
            image_feats = data['image_feats'].cuda().squeeze()
            audio_mfcc = data['audio_mfcc'].cuda().squeeze()
            labels = data['labels'].cuda().squeeze()
            masks = data['masks'].cuda().squeeze()
            video_length = loader.dataset.get_video_length()

            for sec, frames in enumerate(range(0, video_length, 15)):
                torch.cuda.synchronize()
                optimizer.zero_grad()
                img_feats = image_feats[frames:(frames+15)]
                mfcc = audio_mfcc[sec]
                seq_probs, _ = model(img_feats, mfcc, labels, 'train')

                loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
                loss.backward()
                clip_grad_value_(model.parameters(), opt['grad_clip'])
                optimizer.step()
                train_loss = loss.item()
                torch.cuda.synchronize()
                iteration += 1

                print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))

                if epoch % opt["save_checkpoint_every"] == 0:
                    model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
                    model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
                    torch.save(model.state_dict(), model_path)
                    print("model saved to %s" % (model_path))
                    with open(model_info_path, 'a') as f:
                        f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))


def main(opt):
    dataset = VideoAudioDataset(opt, 'train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    opt['vocab_size'] = dataset.get_vocab_size()
    model = MultimodalAtt(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], opt['dim_word'], dim_vid=opt['dim_vid'],
    n_layers=opt['num_layers'], rnn_cell=opt['rnn_type'], rnn_dropout_p=opt['rnn_dropout_p'])
    model = model.cuda()
    crit = LanguageModelCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(loader, model, crit, optimizer, exp_lr_scheduler, opt)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)



