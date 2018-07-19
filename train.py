import torch
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
        save_flag=True
        lr_scheduler.step()
        iteration = 0

        for data in loader:
            image_feats = data['image_feats'].cuda()
            audio_mfcc = data['audio_mfcc'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()

            torch.cuda.synchronize()
            optimizer.zero_grad()
            
            seq_probs, _ = model(image_feats, audio_mfcc, labels, 'train', opt=opt)

            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1

            print("iter %d (epoch %d), train_loss = %.6f" % (iteration, epoch, train_loss))

            if epoch % opt["save_checkpoint_every"] == 0 and not epoch == 0 and save_flag:
                model_path = os.path.join(opt["checkpoint_path"], 'model_%d.pth' % (epoch))
                model_info_path = os.path.join(opt["checkpoint_path"], 'model_score.txt')
                torch.save(model.state_dict(), model_path)
                print("model saved to %s" % (model_path))
                with open(model_info_path, 'a') as f:
                    f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
                save_flag=False


def main(opt):
    dataset = VideoAudioDataset(opt, 'train')
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
    opt['vocab_size'] = dataset.get_vocab_size()
    model = MultimodalAtt(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], opt['dim_word'], dim_vid=opt['dim_vid'],
    n_layers=opt['num_layers'], rnn_dropout_p=opt['rnn_dropout_p'])
    model = model.cuda()
    crit = LanguageModelCriterion()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"],
        amsgrad=True)
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
