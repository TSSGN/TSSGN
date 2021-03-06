from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import h5py
import os
import numpy as np
import random
import torch
import torch.utils.data as data
import six

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """

    def __init__(self, db_path, ext):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(x)
        else:
            self.loader = lambda x: np.load(x)['feat']
        if db_path.endswith('.pth'):  # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        else:
            self.db_type = 'dir'

    def get(self, key):

        if self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key)
            f_input = six.BytesIO(byteflow)
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        else:
            f_input = os.path.join(self.db_path, key + self.ext)

        # load image
        feat = self.loader(f_input)

        return feat

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=1,
                                                 sampler=SubsetSampler(self.dataloader.split_ix[self.split][
                                                                       self.dataloader.iterators[self.split]:]),
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=4,  # 4 is usually enough
                                                 collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[-1] == ix, "ix not equal"

        return tmp + [wrapped]

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # 加载json文件
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
        # 加载h5文件
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # 加载label信息
            seq_size = self.h5_label_file['labels'].shape
            self.label = self.h5_label_file['labels'][:]
            self.seq_length = seq_size[1]
            print('max sequence length in data is', self.seq_length)
            # features extraction
            self.c3d_feature_file = h5py.File(self.opt.input_c3d_feature, 'r', driver='core')
            self.feature_app_file = h5py.File(self.opt.input_app_feature, 'r', driver='core')
            # 加载文件的指针信息
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
            # 加载指示符信息
            self.indicator_status = self.h5_label_file['labels_status'][:]
            # 用于强化学习
            self.label_gts = self.h5_label_file['labels_gts'][:]

        self.num_images = len(self.info['images'])  # self.label_start_ix.shape[0]
        print('read %d sentence features' % (self.num_images))

        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if not 'split' in img:
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        ix1 = self.label_start_ix[ix] - 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
        if ncap < seq_per_img:
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            status = np.zeros([seq_per_img, self.seq_length], dtype='int')
            gts = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
                status[q, :] = self.indicator_status[ixl, :self.seq_length]
                gts[q, :] = self.label_gts[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]
            status = self.indicator_status[ixl: ixl + seq_per_img, :self.seq_length]
            gts = self.label_gts[ixl: ixl + seq_per_img, :self.seq_length]
        return seq, status, gts

    def get_batch(self, split, batch_size = None):
        batch_size = batch_size or self.batch_size
        seq_per_img = self.seq_per_img

        wrapped = False
        fc_batch = []
        label_batch = []
        status_batch = []
        infos = []
        gts = []

        for i in range(batch_size):
            tmp_fc, tmp_token, tmp_status, tmp_gts, ix, tmp_wrapped = self._prefetch_process[split].get()
            if tmp_wrapped:
                wrapped = True
            fc_batch.append(tmp_fc)
            tmp_label = np.zeros([seq_per_img, self.seq_length + 1], dtype='int')  # 只使用<BOS>,不需要<EOS>
            tmp_label_status = np.zeros([seq_per_img, self.seq_length + 1], dtype='int')
            tmp_label_gts = np.zeros([seq_per_img, self.seq_length], dtype='int')  # gt 不需要<BOS>

            tmp_label[:, 1: self.seq_length + 1] = tmp_token
            tmp_label_status[:, 1: self.seq_length + 1] = tmp_status

            tmp_label_gts[:, :] = tmp_gts

            label_batch.append(tmp_label)
            status_batch.append(tmp_label_status)

            gts.append(tmp_gts)
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)

        fc_batch, label_batch, status_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, label_batch, status_batch, gts, infos), key=lambda x: 0, reverse=True))

        data = {}
        max_att_len = max([_.shape[0] for _ in fc_batch])
        data['fc_feats'] = np.zeros([len(fc_batch) * seq_per_img, max_att_len, fc_batch[0].shape[1]], dtype='float32')
        for i in range(len(fc_batch)):
            data['fc_feats'][i * seq_per_img:(i + 1) * seq_per_img, :fc_batch[i].shape[0]] = fc_batch[i]
        data['att_masks'] = None
        data['labels'] = np.vstack(label_batch)
        data['status'] = np.vstack(status_batch)

        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 1], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}
        return data

    def __getitem__(self, index):
        ix = index
        c3d_feature = self.c3d_feature_file[str(self.info['images'][ix]['id'])].value
        fc_feat = self.feature_app_file[str(self.info['images'][ix]['id'])].value
        mot_feat = np.concatenate((fc_feat, c3d_feature), axis=-1)

        seq, status, gts = self.get_captions(ix, self.seq_per_img)

        return (mot_feat, seq, status, gts, ix)

    def __len__(self):
        return len(self.info['images'])