import json
from copy import copy
from queue import Queue
from tqdm import tqdm
import argparse
from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP(r'parser/stanford-corenlp-full-2018-10-05')

def ParseSentence(raw, tokens):
    dependency_tree = nlp.dependency_parse(raw)
    dependency_tree = sorted(dependency_tree, key=lambda a: a[2], reverse=False)
    token_dict = {}
    for index, token in enumerate(tokens):
        token_dict[index] = token
    stack = Queue()
    stack.put(0)
    sorted_dependency_tree = []
    while stack.empty() is False:
        root_token = stack.get()
        for dependency_entry in dependency_tree:
            if dependency_entry[1] == root_token:
                sorted_dependency_tree.append(copy(dependency_entry))
                stack.put(dependency_entry[2])
    assert len(sorted_dependency_tree) == len(tokens)
    format = []
    for index, dependency_entry in enumerate(sorted_dependency_tree):
        type = dependency_entry[0]
        root = dependency_entry[1]
        self = dependency_entry[2]
        self_token = tokens[self - 1]
        left_children = 0
        right_children = 0
        sibling = 0
        for b_index in range(index + 1, len(sorted_dependency_tree)):
            b_dependency_entry = sorted_dependency_tree[b_index]
            if b_dependency_entry[1] == root and (
                    (self < root and b_dependency_entry[2] < root) or (self > root and b_dependency_entry[2] > root)):
                sibling = 1
            if b_dependency_entry[1] == self:
                if b_dependency_entry[2] < self:
                    left_children = 1
                elif b_dependency_entry[2] > self:
                    right_children = 1
        format.append([self_token, left_children, right_children, sibling, type])
    return format

def main(params):
    file = json.load(open(params["input_json"], "r"))
    output = {}
    for sentence in tqdm(file["sentences"]):
        caption = sentence["caption"]
        video_id = sentence["video_id"]
        tokens = nlp.word_tokenize(caption)
        format = ParseSentence(caption, tokens)
        if video_id not in output.keys():
            output[video_id] = []
        video = {}
        video['raw'] = caption
        video['tokens'] = tokens
        video['format'] = format
        output[video_id].append(video)
    split = {}
    for video in file["videos"]:
        video_split = video["split"]
        video_id = video["video_id"]
        split[video_id] = video_split
    videos = []
    for videoid, sentences in output.items():
        tmp = {}
        tmp['filename'] = videoid
        tmp['imgid'] = videoid.split("video")[1]
        tmp['sentences'] = sentences
        tmp['split'] = split[video_id]
        videos.append(tmp)
    with open(params["output_json"], "w") as f:
        json.dump(videos, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", default="data/dataset_msrvtt.json")
    parser.add_argument("--output_json", default="data/dataset.json")

    args = parser.parse_args()
    params = vars(args)
    main(params)

