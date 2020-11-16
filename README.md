## Tree-Structured Sentence Generation Network for Syntax-Aware Video Captioning

This repository includes the implementation for Tree-Strucutred Sentence Generation Network (TSSGN) for Syntax-Aware Video Captioning.

### Requirements
* Python 3.6
* PyTorch 1.0
* Java 1.8.0
* [cider](https://github.com/ruotianluo/cider/tree/e9b736d038d39395fa2259e39342bb876f1cc877)
* [coco-caption](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092)
* [Stanford-CoreNLP 3.9.2](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip)

### Data Preparation


##### Tree Representation

Make sure you have downloaded ``stanford-corenlp-full-2018-10-05``, and put it in ``\parser\``.  
The json file of MSR-VTT, you can find it [here](http://ms-multimedia-challenge.com/2017/dataset).

```bash
python scripts/tree-representation.py --input_json data/dataset_msrvtt.json --ouput_json data/dataset.json
```

##### Label Preparation

```bash
python scripts/prepare_label.py --input_json data/dataset.json --output_json dataset/msrvtttalk.json --output_h5 data/msrvtttalk_label
```

```bash
python scripts/prepare_ngrams.py --input_json data/dataset.json --dict_json data/msrvtttalk.json --output_pkl data/msrvtt-train --split train
```

After data preprocessing, the ``/data/`` should contains six differnt files, i.e.:
* dataset.json
* dataset_msrvtt.json
* msrvtttalk.json
* msrvtttalk_label.h5
* msrvtt-train-idxs.p
* msrvtt-train-words.p

All processed data files are also provided [here](https://drive.google.com/drive/folders/1u6HsgZf9dksKbCQHuPcuHTuOlbMfXMLH).

### Training

##### Training with Cross-Entropy Loss
```bash
python train.py --learning_rate 2e-4 --learning_rate_decay_start 0 --learning_rate_decay_every 2 --learning_rate_decay_rate 0.8 --max_epochs 12 --batch_size 10 --save_checkpoint_every 300 --checkpoint_path log --dataset msrvtt --self_critical_after -1 --input_json data/msrvtttalk.json --input_label_h5 data/msrvtttalk_label.h5 --input_c3d_feature data/msrvtt_c3d_features.h5 --input_app_feature data/msrvtt_appearance_features.h5 --cached_tokens data/msrvtt-train-idxs --caption_model TSSGN
```

##### Training with Self-Critical Loss
```bash
python train.py --learning_rate 2e-5 --learning_rate_decay_start -1 --max_epochs 40 --batch_size 10 --save_checkpoint_every 300 --checkpoint_path log --dataset msrvtt --self_critical_after 0 --input_json data/msrvtttalk.json --input_label_h5 data/msrvtttalk_label.h5 --input_c3d_feature data/msrvtt_c3d_features.h5 --input_app_feature data/msrvtt_appearance_features.h5 --cached_tokens data/msrvtt-train-idxs --start_from log --caption_model TSSGN --reduce_on_plateau
```

The training process can be like:  
![](https://github.com/TSSGN/TSSGN/blob/main/MSR-VTT-Training.png)  

Meanwhile, the input features, i.e., input_c3d_feature and input_app_feature can be downloaded [here]()

### Evaluation
```bash
python eval.py --model log/model-best.pth --infos_path log/infos_-best.pkl --output_json results.json
```

The pretrained models can be downloaded from [here]()  
The coco-type preprocessed dataset of MSR-VTT can also be downloaded from [here]()

### Performance

The performance of TSSGN on MSR-VTT is shown below:  

BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | METEOR | ROUGE
:---: | :---: | :---: | :---: | :---: | :---: | :---: 
81.9|66.8|52.0|38.5|49.7|27.7|61.0

### Acknowledgements

The implementation is based on [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
