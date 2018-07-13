import torch
from torch.utils.data import DataLoader


import NLUtils
from cocoeval import suppress_stdout_stderr, COCOScorer

def eval(model, crit, dataset, vocab, opt):
    model.eval()
    loader = DataLoader(dataset, batch_size=opt['batch_size'], shuffle=True)
    scorer = COCOScorer()