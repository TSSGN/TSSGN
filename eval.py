from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

os.environ["CUDA_VISIBLE_DEVICES"]='1'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='log/model-best.pth')
parser.add_argument('--infos_path', type=str, default='log/infos_-best.pkl')
parser.add_argument('--output_json', type=str, default='results.josn')
parser.add_argument('--input_json', type=str, default='data/msrvtttalk.json')
parser.add_argument('--input_label_h5', type=str, default='data/msrvtttalk_label.h5')
parser.add_argument('--input_c3d_feature', type=str, default='data/msrvtt_c3d_features.h5')
parser.add_argument('--input_app_feature', type=str, default='data/msrvtt_appearance_features.h5')
parser.add_argument('--cached_tokens', type=str, default='data/msrvtt-train-idxs')
opts.add_eval_options(parser)
opt = parser.parse_args()

with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader,  vars(opt))

for k, v in lang_stats.items():
    print('{}: {}'.format(k, v))

with open(opt.output_json, "w") as f:
    json.dump(split_predictions, f)
