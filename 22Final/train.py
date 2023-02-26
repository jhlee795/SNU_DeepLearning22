import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np

from config import cfg, cfg_from_file
import pprint
import datetime
import dateutil.tz
import time
import clip

from models import *
from encoders import CNN_ENCODER, RNN_ENCODER
from dataset import CUBDataset
from utils import *
from loss import *

# Set a config file as 'train_birds.yml' in training, as 'eval_birds.yml' for evaluation
cfg_from_file('cfg/train_birds.yml') # eval_birds.yml

print('Using config:')
pprint.pprint(cfg)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)


imsize = cfg.TREE.BASE_SIZE * (4 ** (cfg.TREE.BRANCH_NUM - 1))
image_transform = transforms.Compose([
    transforms.Resize(int(imsize)),
    transforms.RandomCrop(imsize),
    transforms.RandomHorizontalFlip()])

train_dataset = CUBDataset(cfg.DATA_DIR, transform=image_transform, split='train')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

data = iter(train_dataloader)
data = next(data)

print(data['caps'].shape)
print(f'train data directory:\n{train_dataset.split_dir}')
print(f'# of train filenames:{train_dataset.filenames.shape}')
print(f'example of filename of train image:{train_dataset.filenames[0]}')
print(f'example of caption and its ids:\n{train_dataset.captions[0]}\n{train_dataset.captions_ids[0]}\n')
#print(f'# of train captions:{np.asarray(train_dataset.captions).shape}')
#print(f'# of train caption ids:{np.asarray(train_dataset.captions_ids).shape}')