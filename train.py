import torch.nn as nn
from dataloader import VideoAudioDataset
from torch.utils.data import DataLoader

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
            video_length = data['video_length']

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
    



