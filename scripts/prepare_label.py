import json
import numpy as np
import h5py
import argparse
MaxSentenceLength = 16
WordCountThreshold = 1

def build_vocab(videos):
    counts = {}
    for video in videos:
        for sentence in video["sentences"]:
            for word in sentence["tokens"]:
                counts[word] = counts.get(word, 0) + 1
    bad_words = [w for w, n in counts.items() if n <= WordCountThreshold]
    vocab = [w for w, n in counts.items() if n > WordCountThreshold]
    bad_count = sum(counts[w] for w in bad_words)
    if bad_count > 0:
        vocab.append("UNK")
    for video in videos:
        video["final_format"] = []
        video["final_tokens"] = []
        for sent in video["sentences"]:
            format = sent['format']
            tokens = sent['tokens']
            new_format = []
            for entry in format:
                word = entry[0]
                if counts.get(word, 0) > WordCountThreshold:
                    new_format.append(entry)
                else:
                    word = 'UNK'
                    new_entry = [word, entry[1], entry[2], entry[3], entry[4]]
                    new_format.append(new_entry)
            video['final_format'].append(new_format)
            new_sentence = []
            for token in tokens:
                if counts.get(token, 0) > WordCountThreshold:
                    new_sentence.append(token)
                else:
                    new_sentence.append('UNK')
            video['final_tokens'].append(new_sentence)
    return vocab

def encode_captions(videos, wtoi):
    N = len(videos)
    M = sum(len(video['final_format']) for video in videos)
    gts_arrays = []
    label_arrays = []
    status_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, video in enumerate(videos):
        n = len(video["final_format"])
        assert n > 0
        Li_gts = np.zeros((n, MaxSentenceLength), dtype='uint32')
        Li = np.zeros((n, MaxSentenceLength), dtype='uint32')
        Li_status = np.zeros((n, MaxSentenceLength), dtype='uint32')
        for j, format in enumerate(video["final_format"]):
            label_length[caption_counter] = min(MaxSentenceLength, len(format))
            for k, entry in enumerate(format):
                if k < MaxSentenceLength:
                    Li[j, k] = wtoi[entry[0]]
                    # sibling, leftchild, rightchild -> decimal representation -> section 3.2
                    Li_status[j, k] = entry[3] * 4 + entry[1] * 2 + entry[2] * 1
            tokens = video['final_tokens'][j]
            for k, token in enumerate(tokens):
                if k < MaxSentenceLength:
                    Li_gts[j, k] = wtoi[token]
            caption_counter += 1
        label_arrays.append(Li)
        status_arrays.append(Li_status)
        gts_arrays.append(Li_gts)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1
        counter += n
    L = np.concatenate(label_arrays, axis=0)
    L_gts = np.concatenate(gts_arrays, axis=0)
    assert L.shape[0] == M
    L_status = np.concatenate(status_arrays, axis=0)
    return L, L_gts, L_status, label_start_ix, label_end_ix, label_length

def main(params):
    videos = json.load(open(params["input_json"], "r"))
    vocab = build_vocab(videos)

    itow = {i + 1: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}

    L, L_gts, L_status, label_start_ix, label_end_ix, label_length = encode_captions(videos, wtoi)

    f_lb = h5py.File(params["output_h5"], "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("labels_gts", dtype='uint32', data=L_gts)
    f_lb.create_dataset("labels_status", dtype='uint32', data=L_status)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    out = {}
    out['ix_to_word'] = itow
    out['videos'] = []
    for i, video in enumerate(videos):
        jvideo = {}
        jvideo["split"] = video["split"]
        jvideo["id"] = video["imgid"]
        jvideo["filename"] = video["filename"]
        out['videos'].append(jvideo)
    json.dump(out, open(params["output_json"], "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", default='data/dataset.json')
    parser.add_argument("--output_json", default='data/msrvtttalk.json')
    parser.add_argument("--output_h5", default='data/msrvtttalk_label.h5')

    args = parser.parse_args()
    params = vars(args)
    main(params)