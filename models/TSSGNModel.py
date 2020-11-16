from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from .CaptionModel import CaptionModel

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.att_hid_size = opt.att_hid_size
        self.feat_size = opt.feat_size
        self.gru_size = opt.rnn_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.
        self.opt = opt
        """WORD EMBEDDING"""
        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        """FEATURE EMBEDDING"""
        self.att_feat_size = 1536
        self.c3d_feat_size = 2048
        self.c3d_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(28),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size + self.c3d_feat_size, self.feat_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(28),) if self.use_bn == 2 else ())))
        self.c3d2att = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_state(self, batch_size):
        """Zero Initialize State"""
        weight = next(self.parameters())
        return weight.new_zeros(batch_size, self.rnn_size), weight.new_zeros(batch_size, self.rnn_size)

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats):
        fc_feats = fc_feats.float()
        fc_feats = self.c3d_embed(fc_feats)
        p_fc_feats = self.c3d2att(fc_feats)
        mean_fc_feats = torch.mean(fc_feats, dim=1)
        return mean_fc_feats, fc_feats, p_fc_feats

    ###################################################################

    """FUNCTIONS TO DEAL WITH TREE STRUCTURE"""

    def generate_sibling_mask(self, batch_size, sibling):
        # generate sibling mask [0,1] for previous node, i.e., whether the previous node is the sibling node
        prob = torch.zeros(batch_size, 2).cuda()
        prob_counter = 0
        for ppp in sibling:
            if ppp == 0:
                prob[prob_counter, 0] = 1
            else:
                prob[prob_counter, 1] = 1
            prob_counter += 1
        return prob

    def generate_parent_input_mask(self, batch_size, parent_mask):
        """Generate one-hot parent mask and left-right mask"""
        parent_index = parent_mask.new_zeros(batch_size, dtype=torch.long)
        left_right_index = parent_mask.new_zeros(batch_size, dtype=torch.long)
        parent_input_mask = parent_mask.new_zeros((batch_size, self.seq_length), dtype=torch.float)
        left_right = parent_input_mask.new_zeros((batch_size, 2), dtype=torch.float)
        for batch_id in range(batch_size):
            batch_parent_mask = parent_mask[batch_id, :]
            for i in range(parent_mask.size(1)):
                if batch_parent_mask[i] == 3:
                    """Left and Right Child"""
                    parent_input_mask[batch_id, i] = 1
                    left_right[batch_id, 0] = 1
                    parent_index[batch_id] = i
                    left_right_index[batch_id] = 0
                    break
                elif batch_parent_mask[i] == 2:
                    """Only Left Child"""
                    parent_input_mask[batch_id, i] = 1
                    left_right[batch_id, 0] = 1
                    parent_index[batch_id] = i
                    left_right_index[batch_id] = 0
                    break
                elif batch_parent_mask[i] == 1:
                    """Only Right Child"""
                    parent_input_mask[batch_id, i] = 1
                    left_right[batch_id, 1] = 1
                    parent_index[batch_id] = i
                    left_right_index[batch_id] = 1
                    break
        return parent_input_mask, parent_index, left_right, left_right_index

    def add_parent_mask(self, batch_size, parent_mask, left_child, right_child, index):
        """Adjust parent mask after current time step's generation"""
        for batch_id in range(batch_size):
            left = left_child[batch_id]
            right = right_child[batch_id]
            if left == 1 and right == 1:
                parent_mask[batch_id, index] = 3
            elif left == 1 and right == 0:
                parent_mask[batch_id, index] = 2
            elif left == 0 and right == 1:
                parent_mask[batch_id, index] = 1
            else:
                parent_mask[batch_id, index] = 0
        return parent_mask

    def update_parent_mask(self, batch_size, parent_mask, parent_index, sibling, left_right):
        for batch_id in range(batch_size):
            """index's parent end generation"""
            if sibling[batch_id] == 0:
                parent = parent_index[batch_id]
                if parent_mask[batch_id, parent] == 3:
                    parent_mask[batch_id, parent] = 1
                else:
                    parent_mask[batch_id, parent] = 0
        return parent_mask

    def break_status(self, last_status):
        """Process status to obtain the binary representaion of tree structure"""
        last_has_sibling = (last_status >= 4).float()
        last_status = last_status % 4
        last_has_leftchild = (last_status >= 2).float()
        last_status = last_status % 2
        last_has_rightchild = (last_status >= 1).float()
        return last_has_sibling, last_has_leftchild, last_has_rightchild

    ###################################################################

    def get_logprobs_state(self, sibling, parent, status, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention):
        sibling_xt = self.embed(sibling)
        parent_xt = self.embed(parent)
        """CORE FUNCTION"""
        output, topo_output, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention = \
            self.core(sibling_xt, sibling, parent_xt, parent, status, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention)
        """SOFTMAX PROBABILITY"""
        logprobs = F.log_softmax(output, dim=1)
        logprobs_topo_output = F.log_softmax(topo_output, dim=1)
        """RETURN"""
        return logprobs, logprobs_topo_output, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention

    def sample_logprobs_state(self, sibling, parent, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention, sample_method):
        sibling_xt = self.embed(sibling)
        parent_xt = self.embed(parent)
        """CORE FUNCTION"""
        output, topo_output, topo_logprobs, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention, visualize_weights = \
            self.core.sample(sibling_xt, sibling, parent_xt, parent, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention, sample_method)
        """SOFTMAX PROBABILITY"""
        batch_size = fc_feats.size(0)
        # Get rid of the probabiliry of parent's and sibling's and UNK's word (optional)
        logprobs_masks = fc_feats.new_ones(batch_size, self.vocab_size)
        logprobs_masks[:, output.size(1) - 1] = 0
        for bdash in range(batch_size):
            if sibling[bdash] > 0:
                logprobs_masks[bdash, sibling[bdash] - 1] = 0
            if parent[bdash] > 0:
                logprobs_masks[bdash, parent[bdash] - 1] = 0
        output = output.masked_fill(logprobs_masks == 0, -1e9)
        logprobs = F.log_softmax(output, dim=1)
        """RETURN"""
        return logprobs, topo_output, topo_logprobs, sibling_state, parent_state, sibling_att_out, parent_att_out, visualize_weights

    ###################################################################

    def _forward(self, fc_feats, att_feats, seq, seq_status, att_masks=None):
        batch_size = fc_feats.size(0)
        seq = seq[:, 1:]
        seq_status = seq_status[:, 1:]
        """FEATURE PREPARE"""
        mean_fc_feats, fc_feats, p_fc_feats = self._prepare_feature(fc_feats, att_feats)
        """OUTPUTS"""
        label_outputs = fc_feats.new_zeros(batch_size, self.seq_length, self.vocab_size)
        status_outputs = fc_feats.new_zeros(batch_size, self.seq_length, 8)
        """STATE INITIALIZATION"""
        sibling_state, sibling_att_out = self.init_state(batch_size)
        parent_state, parent_att_out = self.init_state(batch_size)
        """PARENT MASKS"""
        parent_mask = seq.new_zeros(seq.size(), requires_grad=False)
        """PARENT HIDDEN AND ATTENTION OUT SELECTION"""
        parent_hiddens = fc_feats.new_zeros(batch_size, seq.size(1), self.gru_size)
        parent_att_outputs = fc_feats.new_zeros(batch_size, seq.size(1), self.gru_size)
        parent_attentions = fc_feats.new_zeros(batch_size, seq.size(1), self.rnn_size)
        """UNFINISHED"""
        unfinished = fc_feats.new_ones(batch_size)
        """INITIALIZATION SIBLING AND PARENT ATTENTION"""
        sibling_attention = mean_fc_feats
        parent_attention = mean_fc_feats
        """PREDICT THE SEQUENCE"""
        for i in range(seq.size(1)):
            if i == 0:
                sibling = seq.new_zeros(batch_size)
                parent = seq.new_zeros(batch_size)
                left_right = seq.new_zeros((batch_size, 2), dtype=torch.float)
                left_right[:, 0] = 1
            else:
                last_status = seq_status[:, i - 1]
                last_has_sibling, last_has_leftchild, last_has_rightchild = self.break_status(last_status)
                """GENERATE SIBLING INPUT"""
                prob = self.generate_sibling_mask(batch_size, last_has_sibling)
                sibling_input = seq[:, i - 1].clone()
                width_start = sibling_input.new_zeros(sibling_input.size())
                temp_sibling = torch.cat([width_start.unsqueeze(1), sibling_input.unsqueeze(1)], dim=1).float()
                sibling = torch.bmm(prob.unsqueeze(1), temp_sibling.unsqueeze(2)).squeeze(1).squeeze(1).long()
                """GENERATE SIBLING STATE INPUT"""
                start_state, start_sibling_att_out = self.init_state(batch_size)
                temp_sibling_state = torch.cat([start_state.unsqueeze(1), sibling_state.unsqueeze(1)], dim=1)
                temp_sibling_att_out = torch.cat([start_sibling_att_out.unsqueeze(1), sibling_att_out.unsqueeze(1)], dim=1)
                temp_sibling_attention = torch.cat([mean_fc_feats.unsqueeze(1), sibling_attention.unsqueeze(1)], dim=1)
                sibling_state = torch.bmm(prob.unsqueeze(1), temp_sibling_state).squeeze(1)
                sibling_att_out = torch.bmm(prob.unsqueeze(1), temp_sibling_att_out).squeeze(1)
                sibling_attention = torch.bmm(prob.unsqueeze(1), temp_sibling_attention).squeeze(1)
                """GENERATE PARENT INPUT"""
                parent_input_mask, parent_index, left_right, left_right_index = self.generate_parent_input_mask(batch_size, parent_mask)
                parent = torch.bmm(parent_input_mask.unsqueeze(1), seq.float().unsqueeze(2)).long().squeeze(1).squeeze(1)
                """GENERATE PARENT STATE INPUT"""
                parent_state = torch.bmm(parent_input_mask.unsqueeze(1), parent_hiddens.clone()).squeeze(1)
                parent_att_out = torch.bmm(parent_input_mask.unsqueeze(1), parent_att_outputs.clone()).squeeze(1)
                parent_attention = torch.bmm(parent_input_mask.unsqueeze(1), parent_attentions.clone()).squeeze(1)
            """GET LOGPROBS"""
            status = seq_status[:, i]
            has_sibling, has_leftchild, has_rightchild = self.break_status(status)
            logprobs, status_logprobs, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention = \
                self.get_logprobs_state(sibling, parent, status, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention)
            """OUTPUTS SAVE"""
            label_outputs[:, i] = logprobs
            status_outputs[:, i] = status_logprobs
            """UPDATE PARENT MASK"""
            parent_mask = self.add_parent_mask(batch_size, parent_mask, has_leftchild, has_rightchild, i)
            if i > 0:
                parent_mask = self.update_parent_mask(batch_size, parent_mask, parent_index, has_sibling, left_right_index)
            """UPDATE PARENT STATE"""
            parent_hiddens[:, i] = parent_state
            parent_att_outputs[:, i] = parent_att_out
            parent_attentions[:, i] = parent_attention
            """UPDATE ENDING FLAG"""
            unfinished_type = unfinished > 0
            temp_flag_child = has_leftchild + has_rightchild
            temp_flag_sibling = has_sibling - 1
            unfinished = unfinished + temp_flag_child.float() + temp_flag_sibling.float()
            unfinished = unfinished * unfinished_type.type_as(unfinished)
            if unfinished.sum() == 0:
                break
        """RETURN"""
        return label_outputs, status_outputs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)
        batch_size = fc_feats.size(0)
        """FEATURE PREPARE"""
        mean_fc_feats, fc_feats, p_fc_feats = self._prepare_feature(fc_feats, att_feats)
        """OUTPUTS"""
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros((batch_size, self.seq_length))
        seq_left_child = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_right_child = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_sibling = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_status = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_status_logprobs = fc_feats.new_zeros((batch_size, self.seq_length))
        output_visualize_weights = []
        """STATE INITIALIZATION"""
        sibling_state, sibling_att_out = self.init_state(batch_size)
        parent_state, parent_att_out = self.init_state(batch_size)
        """PARENT MASKS"""
        parent_mask = seq.new_zeros(seq.size(), requires_grad=False)
        """PARENT HIDDEN AND ATTENTION OUT SELECTION"""
        parent_hiddens = fc_feats.new_zeros(batch_size, seq.size(1), self.gru_size)
        parent_att_outputs = fc_feats.new_zeros(batch_size, seq.size(1), self.gru_size)
        parent_attentions = fc_feats.new_zeros(batch_size, seq.size(1), self.rnn_size)
        """UNFINISHED"""
        unfinished = fc_feats.new_ones(batch_size)
        """INITIALIZATION SIBLING AND PARENT ATTENTION"""
        sibling_attention = mean_fc_feats
        parent_attention = mean_fc_feats
        """SEQUENCE PREDICTION"""
        for i in range(seq.size(1)):
            if i == 0:
                sibling = seq.new_zeros(batch_size)
                parent = seq.new_zeros(batch_size)
                left_right = seq.new_zeros((batch_size, 2), dtype=torch.float)
                left_right[:, 0] = 1
            else:
                last_status = seq_status[:, i - 1]
                last_has_sibling, last_has_leftchild, last_has_rightchild = self.break_status(last_status)
                """GENERATE SIBLING INPUT"""
                prob = self.generate_sibling_mask(batch_size, last_has_sibling)
                sibling_input = seq[:, i - 1].clone()
                width_start = sibling_input.new_zeros(sibling_input.size())
                temp_sibling = torch.cat([width_start.unsqueeze(1), sibling_input.unsqueeze(1)], dim=1).float()
                sibling = torch.bmm(prob.unsqueeze(1), temp_sibling.unsqueeze(2)).squeeze(1).squeeze(1).long()
                """GENERATE SIBLING STATE INPUT"""
                start_state, start_sibling_att_out = self.init_state(batch_size)
                temp_sibling_state = torch.cat([start_state.unsqueeze(1), sibling_state.unsqueeze(1)], dim=1)
                temp_sibling_att_out = torch.cat([start_sibling_att_out.unsqueeze(1), sibling_att_out.unsqueeze(1)], dim=1)
                temp_sibling_attention = torch.cat([mean_fc_feats.unsqueeze(1), sibling_attention.unsqueeze(1)], dim=1)
                sibling_state = torch.bmm(prob.unsqueeze(1), temp_sibling_state).squeeze(1)
                sibling_att_out = torch.bmm(prob.unsqueeze(1), temp_sibling_att_out).squeeze(1)
                sibling_attention = torch.bmm(prob.unsqueeze(1), temp_sibling_attention).squeeze(1)
                """GENERATE PARENT INPUT"""
                parent_input_mask, parent_index, left_right, left_right_index = self.generate_parent_input_mask(batch_size, parent_mask)
                parent = torch.bmm(parent_input_mask.unsqueeze(1), seq.float().unsqueeze(2)).long().squeeze(1).squeeze(1)
                """GENERATE PARENT STATE INPUT"""
                parent_state = torch.bmm(parent_input_mask.unsqueeze(1), parent_hiddens.clone()).squeeze(1)
                parent_att_out = torch.bmm(parent_input_mask.unsqueeze(1), parent_att_outputs.clone()).squeeze(1)
                parent_attention = torch.bmm(parent_input_mask.unsqueeze(1), parent_attentions.clone()).squeeze(1)
            """GET LOGPROBS"""
            logprobs, topo_output, topo_logprobs, sibling_state, parent_state, sibling_att_out, parent_att_out, visualize_weight = \
                self.sample_logprobs_state(sibling, parent, left_right, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention, sample_method)
            output_visualize_weights.append(visualize_weight)
            """Sample"""
            it, sampleLogprobs = self.sample_next_word(logprobs.clone(), sample_method, temperature)
            it = it + 1
            """SAMPLE CHILD AND SIBLING"""
            has_sibling, has_left_child, has_right_child = self.break_status(topo_output)
            """SAVE"""
            unfinished_type = unfinished > 0
            it = it * unfinished_type.type_as(it)
            seq[:, i] = it
            seqLogprobs[:, i] = sampleLogprobs.view(-1)
            has_left_child = has_left_child * unfinished_type.type_as(has_left_child)
            seq_left_child[:, i] = has_left_child
            has_right_child = has_right_child * unfinished_type.type_as(has_right_child)
            seq_right_child[:, i] = has_right_child
            has_sibling = has_sibling * unfinished_type.type_as(has_sibling)
            seq_sibling[:, i] = has_sibling
            seq_status[:, i] = topo_output
            seq_status_logprobs[:, i] = topo_logprobs.view(-1)
            """UPDATE PARENT MASK"""
            parent_mask = self.add_parent_mask(batch_size, parent_mask, seq_left_child[:, i], seq_right_child[:, i], i)
            if i > 0:
                parent_mask = self.update_parent_mask(batch_size, parent_mask, parent_index, has_sibling, left_right_index)
            """UPDATE PARENT STATE"""
            parent_hiddens[:, i] = parent_state
            parent_att_outputs[:, i] = parent_att_out
            parent_attentions[:, i] = parent_attention
            """UPDATE ENDING FLAG"""
            unfinished_type = unfinished > 0
            temp_flag_child = seq_left_child[:, i] + seq_right_child[:, i]
            temp_flag_sibling = seq_sibling[:, i] - 1
            unfinished = unfinished + temp_flag_child.float() + temp_flag_sibling.float()
            unfinished = unfinished * unfinished_type.type_as(unfinished)
            if unfinished.sum() == 0:
                break
        """RETURN"""
        return seq, seqLogprobs, seq_left_child, seq_right_child, seq_sibling, seq_status, seq_status_logprobs, output_visualize_weights

class TSSGNModel(AttModel):
    def __init__(self, opt):
        super(TSSGNModel, self).__init__(opt)
        print("---Tree-Structured Sentence Generation Network for Syntax-Aware Video Captioning---")
        self.core = TreeCore(opt)

class GlobalAttention(nn.Module):
    def __init__(self, opt):
        super(GlobalAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, roi_feats, p_roi_feats, att_masks=None):
        dot = self.wh(h).unsqueeze(1).expand_as(p_roi_feats) + p_roi_feats
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        if att_masks is not None:
            weight = weight * att_masks
            weight = weight / weight.sum(1, keepdim=True)
        global_feat = torch.bmm(weight.unsqueeze(1), roi_feats).squeeze(1)
        return global_feat, weight

class ContextAttention(nn.Module):
    def __init__(self, opt):
        super(ContextAttention, self).__init__()
        self.wh = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wv = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wa = nn.Linear(opt.att_hid_size, 1)

    def forward(self, h, single_output, context_output, context_mask):
        context_mask = (context_mask > 0).float()
        single_mask = context_mask.new_ones(context_mask.size())
        mask = torch.cat([single_mask.unsqueeze(1), context_mask.unsqueeze(1)], dim=1)

        roi_feats = torch.cat([single_output.unsqueeze(1), context_output.unsqueeze(1)], dim=1)
        p_roi_feats = self.wv(roi_feats)

        dot = self.wh(h).unsqueeze(1).expand_as(p_roi_feats) + p_roi_feats
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        weight = weight * mask
        weight = weight / weight.sum(1, keepdim=True)

        feat = torch.bmm(weight.unsqueeze(1), roi_feats).squeeze(1)
        return feat, weight

class Composition(nn.Module):
    def __init__(self, opt):
        super(Composition, self).__init__()
        self.rnn_size = opt.rnn_size

    def forward(self, roi_feats, context_feat, att_masks=None):
        context_feats = context_feat.unsqueeze(1).expand_as(roi_feats)
        return context_feats - roi_feats

class CompositionAttention(nn.Module):
    def __init__(self, opt):
        super(CompositionAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wh = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wv = nn.Linear(opt.rnn_size, opt.att_hid_size)
        self.wa = nn.Linear(opt.att_hid_size, 1)

        self.compute_relation = Composition(opt)

    def forward(self, h, roi_feats, context_feat, att_masks=None):
        feats = self.compute_relation(roi_feats, context_feat, att_masks)
        feats_ = self.wv(feats)
        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        comp_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)
        return comp_feat, weight

class OutputAttention(nn.Module):
    def __init__(self, opt):
        super(OutputAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.wv = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wh = nn.Linear(self.rnn_size, self.att_hid_size)
        self.wa = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, single_feat, comp_feat):
        feats = torch.stack([single_feat, comp_feat], dim=1)
        feats_ = self.wv(feats)

        dot = self.wh(h).unsqueeze(1).expand_as(feats_) + feats_
        weight = F.softmax(self.wa(torch.tanh(dot)).squeeze(2), dim=1)
        output_feat = torch.bmm(weight.unsqueeze(1), feats).squeeze(1)

        return output_feat, weight

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.gru_size = self.rnn_size
        self.single_attention = GlobalAttention(opt)
        self.context_attention = ContextAttention(opt)
        self.composition_attention = CompositionAttention(opt)
        self.output_attention = OutputAttention(opt)
        self.glu_layer = nn.Sequential(nn.Linear(self.rnn_size + self.gru_size, self.gru_size * 2), nn.GLU())

    def forward(self, h, roi_feats, p_roi_feats, mask, context):
        global_output, global_weight = self.single_attention(h, roi_feats, p_roi_feats, att_masks=None)
        context_output, context_weight = self.context_attention(h, global_output, context, mask)
        comp_output, comp_weight = self.composition_attention(h, roi_feats, context_output)
        output, output_weight = self.output_attention(h, global_output, comp_output)
        x = self.glu_layer(torch.cat([output, h], dim=-1))
        weights = {}
        weights['global_weight'] = global_weight
        weights['context_weight'] = context_weight
        weights['comp_weight'] = comp_weight
        weights['output_weight'] = output_weight
        return x, output, weights

class TreeCore(nn.Module):
    def __init__(self, opt):
        super(TreeCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm
        self.opt = opt
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        """RNN"""
        self.left_depth_gru = nn.GRUCell(opt.input_encoding_size + opt.rnn_size * 3, opt.rnn_size)
        self.left_depth_norm = nn.LayerNorm(opt.rnn_size)
        self.right_depth_gru = nn.GRUCell(opt.input_encoding_size + opt.rnn_size * 3, opt.rnn_size)
        self.right_depth_norm = nn.LayerNorm(opt.rnn_size)
        self.width_gru = nn.GRUCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size)
        self.width_norm = nn.LayerNorm(opt.rnn_size)
        """Attention"""
        self.left_attention = Attention(opt)
        self.right_attention = Attention(opt)
        self.sibling_attention = Attention(opt)
        """Topology"""
        self.pred_topo = nn.Linear(opt.rnn_size, 8)
        """Learnable Offset Parameters"""
        self.offset_embed = nn.Sequential(
            nn.Embedding(8, self.rnn_size // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))
        """Softmax Layer"""
        self.logit = nn.Linear(opt.rnn_size + opt.rnn_size // 2, opt.vocab_size)

    def forward(self, sibling_xt, sibling, parent_xt, parent, prob_status, child_prob, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention):
        batch_size = fc_feats.size(0)
        """SIBLING INPUT"""
        sibling_gru_input = torch.cat([sibling_xt, sibling_att_out, mean_fc_feats], dim=1)
        """SIBLING"""
        h_width = self.width_gru(sibling_gru_input, sibling_state)
        h_width = self.width_norm(h_width)
        att_width, sibling_attention, _ = self.sibling_attention(h_width, fc_feats, p_fc_feats, sibling, sibling_attention)
        width_output = att_width
        """PARENT INPUT"""
        parent_input = torch.cat([parent_xt, parent_att_out, att_width, mean_fc_feats], dim=-1)
        """LEFT GRU"""
        left_h_depth = self.left_depth_gru(parent_input, parent_state)
        left_h_depth = self.left_depth_norm(left_h_depth)
        left_att, left_parent_attention, _ = self.left_attention(left_h_depth, fc_feats, p_fc_feats, parent, parent_attention)
        """RIGHT GRU"""
        right_h_depth = self.right_depth_gru(parent_input, parent_state)
        right_h_depth = self.right_depth_norm(right_h_depth)
        right_att, right_parent_attention, _ = self.right_attention(right_h_depth, fc_feats, p_fc_feats, parent, parent_attention)
        """LEFT or RIGHT"""
        temp_h_depth = torch.cat([left_h_depth.unsqueeze(1), right_h_depth.unsqueeze(1)], dim=1)
        tmp_att_depth = torch.cat([left_att.unsqueeze(1), right_att.unsqueeze(1)], dim=1)
        tmp_attention = torch.cat([left_parent_attention.unsqueeze(1), right_parent_attention.unsqueeze(1)], dim=1)
        h_depth = torch.bmm(child_prob.unsqueeze(1), temp_h_depth).squeeze(1)
        att_depth = torch.bmm(child_prob.unsqueeze(1), tmp_att_depth).squeeze(1)
        parent_attention = torch.bmm(child_prob.unsqueeze(1), tmp_attention).squeeze(1)
        """PREDICTION"""
        depth_output = att_depth
        att_pred_output = F.dropout(depth_output, self.drop_prob_lm, self.training)
        """TOPOLOGY PREDICTION"""
        topo_output = self.pred_topo(att_pred_output)
        """LANGUAGE GENERATION"""
        offset_embed = self.offset_embed(prob_status)
        output = self.logit(torch.cat([att_pred_output, offset_embed], dim=-1))
        """STATE"""
        sibling_state = h_width
        parent_state = h_depth
        """RETURN"""
        return output, topo_output, sibling_state, parent_state, width_output, depth_output, sibling_attention, parent_attention

    def sample(self, sibling_xt, sibling, parent_xt, parent, child_prob, mean_fc_feats, fc_feats, p_fc_feats, sibling_state, parent_state, sibling_att_out, parent_att_out, sibling_attention, parent_attention, sample_method):
        batch_size = fc_feats.size(0)
        """SIBLING INPUT"""
        sibling_gru_input = torch.cat([sibling_xt, sibling_att_out, mean_fc_feats], dim=1)
        """SIBLING"""
        h_width = self.width_gru(sibling_gru_input, sibling_state)
        h_width = self.width_norm(h_width)
        att_width, sibling_attention, sibling_weight = self.sibling_attention(h_width, fc_feats, p_fc_feats, sibling, sibling_attention)
        width_output = att_width
        """PARENT INPUT"""
        parent_input = torch.cat([parent_xt, parent_att_out, att_width, mean_fc_feats], dim=-1)
        """LEFT GRU"""
        left_h_depth = self.left_depth_gru(parent_input, parent_state)
        left_h_depth = self.left_depth_norm(left_h_depth)
        left_att, left_parent_attention, left_weight = self.left_attention(left_h_depth, fc_feats, p_fc_feats, parent, parent_attention)
        """RIGHT GRU"""
        right_h_depth = self.right_depth_gru(parent_input, parent_state)
        right_h_depth = self.right_depth_norm(right_h_depth)
        right_att, right_parent_attention, right_weight = self.right_attention(right_h_depth, fc_feats, p_fc_feats, parent, parent_attention)
        """LEFT or RIGHT"""
        temp_h_depth = torch.cat([left_h_depth.unsqueeze(1), right_h_depth.unsqueeze(1)], dim=1)
        tmp_att_depth = torch.cat([left_att.unsqueeze(1), right_att.unsqueeze(1)], dim=1)
        tmp_attention = torch.cat([left_parent_attention.unsqueeze(1), right_parent_attention.unsqueeze(1)], dim=1)
        h_depth = torch.bmm(child_prob.unsqueeze(1), temp_h_depth).squeeze(1)
        att_depth = torch.bmm(child_prob.unsqueeze(1), tmp_att_depth).squeeze(1)
        parent_attention = torch.bmm(child_prob.unsqueeze(1), tmp_attention).squeeze(1)
        """PREDICTION"""
        depth_output = att_depth
        att_pred_output = F.dropout(depth_output, self.drop_prob_lm, self.training)
        """TOPOLOGY PREDICTION"""
        topo_output = self.pred_topo(att_pred_output)
        logprobs_topo_output = F.log_softmax(topo_output, dim=1)
        topo_it, topo_logprobs = self.sample_next_word(logprobs_topo_output, sample_method)
        """LANGUAGE GENERATION"""
        offset_embed = self.offset_embed(topo_it)
        output = self.logit(torch.cat([att_pred_output, offset_embed], dim=-1))
        """STATE"""
        sibling_state = h_width
        parent_state = h_depth
        """WEIGHT VISUALIZE"""
        visualize_weights = {}
        visualize_weights['sibling_weight'] = sibling_weight
        visualize_weights['left_weight'] = left_weight
        visualize_weights['right_weight'] = right_weight
        """RETURN"""
        return output, topo_it, topo_logprobs, sibling_state, parent_state, width_output, depth_output, sibling_attention, parent_attention, visualize_weights

    def sample_next_word(self, logprobs, sample_method, temperature=1.0):
        if sample_method == 'greedy':
            sampleLogprobs, it = torch.max(logprobs, 1)
            it = it.view(-1).long()
        else:
            logprobs = logprobs / temperature
            it = torch.distributions.Categorical(logits=logprobs.detach()).sample()
            sampleLogprobs = logprobs.gather(1, it.unsqueeze(1))  # gather the logprobs at sampled positions
        return it, sampleLogprobs