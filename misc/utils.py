from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import six
from six.moves import cPickle
from queue import Queue

bad_endings = ['with','in','on','of','a','at','to','for','an','this','his','her','that']
bad_endings += ['the']

def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)

def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def if_use_feat(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc', 'newfc']:
        use_att, use_fc = False, True
    elif caption_model == 'language_model':
        use_att, use_fc = False, False
    elif caption_model in ['topdown', 'aoa']:
        use_fc, use_att = True, True
    else:
        use_att, use_fc = True, False
    return use_fc, use_att

def RestoreSentence(format):
    if len(format) == 0:
        return []
    tokens = [0]
    root_queue = Queue()
    root_index = 0
    # init
    root_queue.put(0)
    while root_queue.empty() is False:
        root = root_queue.get()
        for token, value in enumerate(tokens):
            if value == root:
                root_index = token
                break
        left_child = format[root][1]
        right_child = format[root][2]
        sibling = format[root][3]
        for index in range(0, len(format)):
            if index in tokens:
                continue
            if left_child == 1:
                tokens.insert(root_index, index)
                root_index += 1
                if format[index][3] == 0:
                    left_child = 0
            elif right_child == 1:
                tokens.insert(root_index + 1, index)
                root_index += 1 ## revised by ZY, 20/2/3, for correct insertion
                if format[index][3] == 0:
                    right_child = 0
            if format[index][1] == 1 or format[index][2] == 1:
                root_queue.put(index)
            if left_child == 0 and right_child == 0:
                break
    sentence = []
    for token in tokens:
        sentence.append(format[token][0])
    return sentence

def restore_sentence(tokens, leftchilds, rightchilds, siblings):
    N, D = tokens.size()
    tokens = tokens.data.cpu().numpy()
    leftchilds = leftchilds.data.cpu().numpy()
    rightchilds = rightchilds.data.cpu().numpy()
    siblings = siblings.data.cpu().numpy()
    out = []
    for i in range(N):
        curr_tokens = tokens[i,:]
        curr_leftchilds = leftchilds[i,:]
        curr_rightchilds = rightchilds[i,:]
        curr_siblings = siblings[i,:]
        format = []
        for index, token in enumerate(curr_tokens):
            if token == 0:
                break
            format.append([token, curr_leftchilds[index], curr_rightchilds[index], curr_siblings[index]])
        sentence = RestoreSentence(format)
        while len(sentence) < D:
            sentence.append(0)
        out.append(sentence)
    out = torch.from_numpy(np.array(out))
    return out

def decode_sequence(ix_to_word, tokens, leftchilds, rightchilds, siblings, new_weights):
    N, D = tokens.size()
    out = []
    token_out = []
    weights = []
    for i in range(N):
        curr_tokens = tokens[i,:]
        curr_leftchilds = leftchilds[i,:]
        curr_rightchilds = rightchilds[i,:]
        curr_siblings = siblings[i,:]
        format = []
        for index, token in enumerate(curr_tokens):
            if token == 0:
                break
            format.append([token, curr_leftchilds[index], curr_rightchilds[index], curr_siblings[index]])
        sentence = RestoreSentence(format)
        txt = ''
        txt2 = ''
        for j in range(len(sentence)):
            if j >= 1:
                txt += ' '
                txt2 += ' '
            txt += ix_to_word[str(sentence[j].item())]
            txt2 += ix_to_word[str(curr_tokens[j].item())]
            weights.append(new_weights[j])
        out.append(txt)
        token_out.append(txt2)
    return out, weights, token_out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class BinearCrossEntropyCriterion(nn.Module):
    def __init__(self):
        super(BinearCrossEntropyCriterion, self).__init__()
        self.loss = nn.BCELoss(reduce=False)

    def forward(self, input, target, mask, reward = None):

        input = input.squeeze() # [80, 31]

        if reward is None:
            output = self.loss(input, target.float()) * mask
        else:
            mask = (mask > 0).float().view(-1)
            # mask = to_contiguous(mask.new(mask.size(0), 1).fill_(1)).view(-1)
            output = self.loss(input, target.float()).view(-1) * reward.view(-1) * mask

        output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, seq_mask):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq_mask>0).float().view(-1)
        # mask = to_contiguous(mask.new(mask.size(0), 1).fill_(1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        target = target - 1
        target = (target.int() * (target > 0).int()).long()
        mask = mask.float()

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterionToken(nn.Module):
    def __init__(self):
        super(LanguageModelCriterionToken, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        input = to_contiguous(input).view(-1, input.size(-1))
        target = to_contiguous(target).view(-1)
        mask = to_contiguous(mask).view(-1)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon,
                             weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon,
                          weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)

def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length

class NoamOpt(object):
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

class ReduceLROnPlateau(object):
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold,
                                                              threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)

    def step(self):
        "Update parameters and rate"
        self.optimizer.step()

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr': self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr)  # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

def get_std_opt(model, factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    return NoamOpt(model.model.tgt_embed[0].d_model, factor, warmup,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

