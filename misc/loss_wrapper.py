import torch
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

class LossWrapper(torch.nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()
        self.opt = opt
        self.model = model
        self.crit = utils.LanguageModelCriterion()
        self.rl_crit = utils.RewardCriterion()
        self.crit_tokens = utils.LanguageModelCriterionToken()

    def forward(self, fc_feats, att_feats, mfcc_feats, category_feats, labels, masks, status, att_masks, gts, gt_indices, sc_flag):
        out = {}
        if not sc_flag:

            outputs, status_outputs = self.model(fc_feats, att_feats, labels, status, att_masks)
            token_loss = self.crit(outputs, labels[:, 1:], masks[:, 1:])
            status_loss = self.crit_tokens(status_outputs, status[:, 1:], masks[:, 1:])
            loss = token_loss + status_loss

            out['token_loss'] = token_loss
            out['status_loss'] = status_loss

        else:
            self.model.eval()
            with torch.no_grad():
                greedy_seq, _, greedy_left_child, greedy_right_child, greedy_sibling, greedy_status, status_logprobs, _ \
                    = self.model(fc_feats, att_feats, att_masks, mode='sample')
                greedy_res = utils.restore_sentence(greedy_seq, greedy_left_child, greedy_right_child, greedy_sibling)
            self.model.train()

            gen_seq, sample_logprobs, gen_left_child, gen_right_child, gen_sibling, gen_status, gen_status_logprobs, status_outputs \
                = self.model(fc_feats, att_feats, att_masks, opt={'sample_method': 'sample'}, mode='sample')
            gen_result = utils.restore_sentence(gen_seq, gen_left_child, gen_right_child, gen_sibling)

            gts = [gts[_] for _ in gt_indices.tolist()]

            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).float().cuda()

            loss = self.rl_crit(sample_logprobs, gen_seq.data, reward, gen_seq.data)
            status_loss = self.rl_crit(gen_status_logprobs, gen_status.data, reward, gen_seq.data)

            out['token_loss'] = loss
            out['status_loss'] = status_loss
            loss = loss + status_loss
            out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out
