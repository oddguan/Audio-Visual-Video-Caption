import torch
import torch.nn as nn


def decode_sequence(ix_to_word, seq):
    seq = seq.cpu()
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j].item()
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
            logits: shape of (N, seq_len, vocab_size)
            target: shape of (N, seq_len)
            mask: shape of (N, seq_len)
        """
        print(mask.shape)
        print(target.shape)
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        print(mask.shape)
        print(target.shape)
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        print(logits.shape)
        print(mask.shape)
        print(target.shape)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output
