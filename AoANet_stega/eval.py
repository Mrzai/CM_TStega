from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from models.AoAModel import AoAModel
from models.SecretModel import SecretEncoder, SecretExtractor
import sys
# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model_path', type=str, default='',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--secret_size', type=int, default=10,
                help='the number of secret bit')
opts.add_eval_options(parser)

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
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
cap_model = AoAModel(opt).cuda()
sec_encoder = SecretEncoder(opt.secret_size).cuda()
sec_extractor = SecretExtractor(opt).cuda()
del opt.vocab

cap_model.load_state_dict(torch.load(os.path.join(opt.model_path, 'cap_model-best.pth')))
sec_encoder.load_state_dict(torch.load(os.path.join(opt.model_path, 'sec_encoder-best.pth')))
sec_extractor.load_state_dict(torch.load(os.path.join(opt.model_path, 'sec_extractor-best.pth')))

crit = utils.LanguageModelCriterion()

# Create the Data Loader instance

loader = DataLoader(opt)

# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
opt.datset = opt.input_json

loss, split_predictions, lang_stats = eval_utils.eval_split(cap_model, sec_encoder, sec_extractor, crit, loader, vars(opt))

print('loss: ', loss)
if lang_stats:
  print(lang_stats)
