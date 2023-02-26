import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from copy import deepcopy
import os

from config import cfg

def my_collate_fn(samples):

    imgs = list()
    imgs.append(torch.stack([sample['img'][0] for sample in samples], dim=0))
    imgs.append(torch.stack([sample['img'][1] for sample in samples], dim=0))
    imgs.append(torch.stack([sample['img'][2] for sample in samples], dim=0))
    clip_enc = torch.stack([sample['clip_enc'] for sample in samples], dim=0)

    caps = torch.from_numpy(np.array([sample['caps'] for sample in samples]))
    cap_len = torch.from_numpy(np.array([sample['cap_len'] for sample in samples]))
    cls_id = torch.from_numpy(np.array([sample['cls_id'] for sample in samples]))
    key = [sample['key'] for sample in samples]
    sent_ix = torch.from_numpy(np.array([sample['sent_ix'] for sample in samples]))
    bert_enc = torch.stack([torch.tensor(sample['bert_enc']) for sample in samples], dim=0)
    bert_mask = torch.stack([torch.tensor(sample['bert_mask']) for sample in samples], dim=0)

    data = {'img': imgs, 'bert_enc': bert_enc, 'bert_mask': bert_mask, 'clip_enc': clip_enc, 'caps': caps, 'cap_len': cap_len, 'cls_id': cls_id, 'key': key, 'sent_ix': sent_ix}
    return data


def my_collate_fn_test(samples):

    clip_enc = torch.stack([sample['clip_enc'] for sample in samples], dim=0)
    caps = torch.from_numpy(np.array([sample['caps'] for sample in samples]))
    cap_len = torch.from_numpy(np.array([sample['cap_len'] for sample in samples]))
    cls_id = torch.from_numpy(np.array([sample['cls_id'] for sample in samples]))
    key = [sample['key'] for sample in samples]
    sent_ix = torch.from_numpy(np.array([sample['sent_ix'] for sample in samples]))
    bert_enc = torch.stack([torch.tensor(sample['bert_enc']) for sample in samples], dim=0)
    bert_mask = torch.stack([torch.tensor(sample['bert_mask']) for sample in samples], dim=0)

    data = {'bert_enc': bert_enc, 'bert_mask': bert_mask, 'clip_enc':clip_enc,'caps': caps,
            'cap_len': cap_len, 'cls_id': cls_id, 'key': key, 'sent_ix': sent_ix}
    return data



def weights_init(m):
    # orthogonal_
    # xavier_uniform_(
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print(m.state_dict().keys())
        if list(m.state_dict().keys())[0] == 'weight':
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif list(m.state_dict().keys())[3] == 'weight_bar':
            nn.init.orthogonal_(m.weight_bar.data, 1.0)
        #nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        

def prepare_labels(batch_size):
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()

    return real_labels, fake_labels, match_labels


def prepare_data(data):
    """
    Prepares data given by dataloader
    e.g., x = Variable(x).cuda()
    """
    bert_enc = data['bert_enc']
    bert_mask = data['bert_mask']
    clip_enc = data['clip_enc']
    captions = data['caps']
    captions_lens = data['cap_len']
    class_ids = data['cls_id']
    keys = data['key']
    sentence_idx = data['sent_ix']

    # sort data by the length in a decreasing order
    # the reason of sorting data can be found in https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html

    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = []
    if 'img' in data.keys():
        imgs = data['img']
        for i in range(len(imgs)):
            imgs[i] = imgs[i][sorted_cap_indices]
            if cfg.CUDA:
                real_imgs.append(Variable(imgs[i]).cuda())
            else:
                real_imgs.append(Variable(imgs[i]))


    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    keys = [keys[i] for i in sorted_cap_indices.numpy()]

    if cfg.CUDA:
        bert_enc = Variable(bert_enc).cuda()
        bert_mask = Variable(bert_mask).cuda()
        clip_enc = Variable(clip_enc).cuda()
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        bert_enc = Variable(bert_enc)
        bert_mask = Variable(bert_mask)
        clip_enc = Variable(clip_enc)
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, bert_enc, bert_mask, clip_enc, captions, sorted_cap_lens, class_ids, keys, sentence_idx]


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)
 
def save_model(netG, text_encoder, image_encoder, model_name='final_model.pth'):
    """
    Saves models
    """
    torch.save({'netG': netG.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'image_encoder': image_encoder.state_dict()},
                os.path.join(cfg.CHECKPOINT_DIR, model_name))