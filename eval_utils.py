from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import json
import os
import misc.utils as utils

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val_msrvtt.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', '.cache_'+ model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', False)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 0)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = 1
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(remove_bad_endings)

    model.eval()
    loader.reset_iterator(split)
    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    results = []

    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [_.cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_masks = tmp

        with torch.no_grad():
            eval_kwargs['type'] = 'eval'
            # Only when val_beam_size == 1, visualize weights make senses
            tokens, token_logprobs, leftchilds, rightchilds, siblings, _, _, visualize_weights = model(fc_feats, att_masks, opt=eval_kwargs, mode='sample')

        new_weights = []
        for weight in visualize_weights:
            temp = {}
            for key, value in weight.items():
                temptemp = {}
                for keykey, valuevalue in value.items():
                    temptemp[keykey] = valuevalue.cpu().detach().numpy().tolist()
                temp[key] = temptemp
            new_weights.append(temp)

        sents, weights, token_sents = utils.decode_sequence(loader.get_vocab(), tokens, leftchilds, rightchilds, siblings, new_weights)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}

            predictions.append(entry)

            result = {"image_id": data['infos'][k]['id'],
                      "caption": sent,
                      "token_sent": token_sents[k],
                      "leftchilds": leftchilds[k].cpu().detach().numpy().tolist(),
                      "rightchilds": rightchilds[k].cpu().detach().numpy().tolist(),
                      "siblings": siblings[k].cpu().detach().numpy().tolist(),
                      "weights": weights,
                      "new_weights": new_weights}

            results.append(result)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' % (ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['checkpoint_path'].split('/')[-1], split)

    model.train()
    return loss_sum / loss_evals, results, lang_stats

