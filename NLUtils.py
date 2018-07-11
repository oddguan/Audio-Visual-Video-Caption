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
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)

    def forward(self, inputs, target, mask):
        """
            inputs: shape of (N, seq_len, vocab_size)
            target: shape of (N, seq_len)
            mask: shape of (N, seq_len)
        """
        batch_size = inputs.shape[0]
        target = target[:, :inputs.shape[1]]
        mask = mask[:, :inputs.shape[1]]
        inputs = inputs.contiguous().view(-1, inputs.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(inputs, target)
        output = torch.sum(loss * mask) / batch_size
        return output
