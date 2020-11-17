import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--learning_rate_decay_every', type=int, default=2)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8)

    parser.add_argument('--use_bn', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=16)

    parser.add_argument('--max_epochs', type=int, default=12)
    parser.add_argument('--seq_per_img', type=int, default=20)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--save_checkpoint_every', type=int, default=300)

    parser.add_argument('--checkpoint_path', type=str, default='log')
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--val_split', type=str, default='test')

    parser.add_argument('--self_critical_after', type=int, default=-1)
    parser.add_argument('--start_from', type=str, default=None)

    parser.add_argument('--input_json', type=str, default='data/msrvtttalk.json')
    parser.add_argument('--input_label_h5', type=str, default='data/msrvtttalk_label.h5')
    parser.add_argument('--input_c3d_feature', type=str, default='data/msrvtt_c3d_features.h5')
    parser.add_argument('--input_app_feature', type=str, default='data/msrvtt_appearance_features.h5')
    parser.add_argument('--cached_tokens', type=str, default='data/msrvtt-train-idxs')

    # Model settings
    parser.add_argument('--caption_model', type=str, default="TSSGN")
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--feat_size', type=int, default=1024)
    parser.add_argument('--input_encoding_size', type=int, default=512)
    parser.add_argument('--att_hid_size', type=int, default=512)

    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)

    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--length_penalty', type=str, default='')
    parser.add_argument('--block_trigrams', type=int, default=0)
    parser.add_argument('--remove_bad_endings', type=int, default=0)

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--optim_alpha', type=float, default=0.9)
    parser.add_argument('--optim_beta', type=float, default=0.999)
    parser.add_argument('--optim_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--noamopt', action='store_true')
    parser.add_argument('--noamopt_warmup', type=int, default=2000)
    parser.add_argument('--noamopt_factor', type=float, default=1)
    parser.add_argument('--reduce_on_plateau', action='store_true')
    parser.add_argument('--use_warmup', type=int, default=0)

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1)
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=2)
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=-1)
    parser.add_argument('--save_history_ckpt', type=int, default=0)
    parser.add_argument('--language_eval', type=int, default=1)
    parser.add_argument('--losses_log_every', type=int, default=25)
    parser.add_argument('--load_best_score', type=int, default=1)

    # misc
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--train_only', type=int, default=0)

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1)
    parser.add_argument('--bleu_reward_weight', type=float, default=0)

    args = parser.parse_args()

    return args

def add_eval_options(parser):
    parser.add_argument('--batch_size', type=int, default=10, help='if > 0 then overrule, otherwise load from checkpoint.')
    parser.add_argument('--num_images', type=int, default=-1, help='how many images to use when periodically evaluating the loss? (-1 = all)')
    parser.add_argument('--language_eval', type=int, default=1, help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--dump_images', type=int, default=0, help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
    parser.add_argument('--dump_json', type=int, default=0, help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--dump_path', type=int, default=0, help='Write image paths along with predictions into vis json? (1=yes,0=no)')

    # Sampling options
    parser.add_argument('--sample_method', type=str, default='greedy', help='greedy; sample; gumbel; top<int>, top<0-1>')
    parser.add_argument('--beam_size', type=int, default=1,  help='indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    parser.add_argument('--max_length', type=int, default=16, help='Maximum length during sampling')
    parser.add_argument('--length_penalty', type=str, default='', help='wu_X or avg_X, X is the alpha')
    parser.add_argument('--group_size', type=int, default=1, help='used for diverse beam search. if group_size is 1, then it\'s normal beam search')
    parser.add_argument('--diversity_lambda', type=float, default=0.5, help='used for diverse beam search. Usually from 0.2 to 0.8. Higher value of lambda produces a more diverse list')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature when sampling from distributions (i.e. when sample_method = sample). Lower = "safer" predictions.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='If 1, not allowing same word in a row')
    parser.add_argument('--block_trigrams', type=int, default=0, help='block repeated trigram.')
    parser.add_argument('--remove_bad_endings', type=int, default=0, help='Remove bad endings')
    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='', help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--image_root', type=str, default='', help='In case the image paths have to be preprended with a root path to an image folder')
    # misc
    parser.add_argument('--id', type=str, default='', help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
    parser.add_argument('--verbose_beam', type=int, default=0, help='if we need to print out all beam search beams.')
    parser.add_argument('--verbose_loss', type=int, default=0, help='If calculate loss using ground truth during evaluation')
