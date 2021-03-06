#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class DecoderWithAttention(nn.Module):
    """
    forward
        input: [batch_size, 25, 512]
        output:
            probability sequence: [batch_size x max_len_labels x num_classes]
            num_classes is the size of vocabulary
    """

    def __init__(self, num_classes, in_planes, sDim, attDim, max_len_labels, use_bidecoder, 
                 device='cuda'
                ):
        super().__init__()
        self.num_classes = num_classes # this is the output classes. So it includes the <EOS>.
        self.in_planes = in_planes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.use_bidecoder = use_bidecoder

        self.decoder = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim)
        if use_bidecoder:
            self.decoder_reverse = DecoderUnit(sDim=sDim, xDim=in_planes, yDim=num_classes, attDim=attDim)
        self.device = device

    def forward(self, x, targets, max_length = 0):
        batch_size = x.size(0)
        if max_length>0:
            max_length = min(max_length, self.max_len_labels)
        else:
            max_length = self.max_len_labels
        if self.use_bidecoder:
            targets, targets_reverse = targets
            
            # Decoder
            state = torch.zeros(1, batch_size, self.sDim, device = self.device)
            state_reverse = torch.zeros(1, batch_size, self.sDim, device = self.device)
            outputs = []
            outputs_reverse = []

            for i in range(max_length):
                if i == 0:
                    y_prev = torch.zeros((batch_size), device = self.device).fill_(self.num_classes) # the last one is used as the <BOS>.
                    y_prev_reverse = torch.zeros((batch_size), device = self.device).fill_(self.num_classes) # the last one is used as the <BOS>.
                else:
                    y_prev = targets[:,i-1]
                    y_prev_reverse = targets_reverse[:,i-1]

                output, state = self.decoder(x, state, y_prev)
                output_reverse, state_reverse = self.decoder_reverse(x, state_reverse, y_prev_reverse)
                outputs.append(output)
                outputs_reverse.append(output_reverse)
            outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
            outputs_reverse = torch.cat([_.unsqueeze(1) for _ in outputs_reverse], 1)
            return outputs, outputs_reverse
        else:
            # Decoder
            state = torch.zeros(1, batch_size, self.sDim, device = self.device)
            outputs = []

            for i in range(max_length):
                if i == 0:
                    y_prev = torch.zeros((batch_size)).fill_(self.num_classes, device = self.device) # the last one is used as the <BOS>.
                else:
                    y_prev = targets[:,i-1]

                output, state = self.decoder(x, state, y_prev)
                outputs.append(output)
            outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)
        return outputs

    def beam_search(self, x, beam_width, eos):
        p, s = self.beam_search_(x, beam_width, eos, self.decoder)
        if not self.use_bidecoder:
            p = p[:,0,:]
            return p, torch.ones_like(p)
        p_r, s_r = self.beam_search_(x, beam_width, eos, self.decoder_reverse)
        for i in range(x.size(0)):
            idx = p_r[i].eq(eos).nonzero()
            if idx.numel() > 0:
                p_r[i][:idx[0][0]] = p_r[i][:idx[0][0]].flip(0)
            else:
                p_r[i] = p_r[i].flip(0)
        s, index = torch.stack((s,s_r)).max(0)
        p = torch.stack((p,p_r),1)
        p = p.gather(1,index.unsqueeze(1).repeat(1,p.size(-1)).unsqueeze(1)).squeeze()
        return p, torch.ones_like(p)
    
    def beam_search2(self, x, beam_width, eos):
        p, s = self.beam_search_(x, beam_width, eos, self.decoder)
        if not self.use_bidecoder:
            p = p[:,0,:]
            return p, torch.ones_like(p)
        p_r, s_r = self.beam_search_(x, beam_width, eos, self.decoder_reverse)
        s, index = torch.stack((s,s_r)).max(0)
        for i in range(x.size(0)):
            if index[i].item()==0:
                continue
            idx = p[i].eq(eos).nonzero()
            if idx.numel() > 0:
                p[i][:idx[0][0]] = p_r[i][:idx[0][0]].flip(0)
            else:
                idx = p_r[i].eq(eos).nonzero()
                if idx.numel() > 0: p[i][:idx[0][0]] = p_r[i][:idx[0][0]].flip(0)
                else: p[i] = p_r[i].flip(0)
        return p, torch.ones_like(p)
        
    def beam_search_(self, x, beam_width, eos, decoder):

        def _inflate(tensor, times, dim):
            repeat_dims = [1] * tensor.dim()
            repeat_dims[dim] = times
            return tensor.repeat(*repeat_dims)

        # https://github.com/IBM/pytorch-seq2seq/blob/fede87655ddce6c94b38886089e05321dc9802af/seq2seq/models/TopKDecoder.py
        batch_size, l, d = x.size()
        # inflated_encoder_feats = _inflate(encoder_feats, beam_width, 0) # ABC --> AABBCC -/-> ABCABC
        inflated_encoder_feats = x.unsqueeze(1).permute((1,0,2,3)).repeat(
            (beam_width,1,1,1)).permute((1,0,2,3)).contiguous().view(-1, l, d)

        # Initialize the decoder
        state = torch.zeros(1, batch_size * beam_width, self.sDim, device = self.device)
        pos_index = (torch.tensor(range(batch_size), device = self.device) * beam_width).long().view(-1, 1)

        # Initialize the scores
#         sequence_scores = torch.Tensor(batch_size * beam_width, 1)
#         sequence_scores.fill_(-float('Inf'))
        sequence_scores = -float('Inf')*torch.ones(batch_size * beam_width, 1, device = self.device)
        sequence_scores.index_fill_(0, torch.tensor(
            [i * beam_width for i in range(0, batch_size)], 
            device = self.device
            ).long(), 0.0)
        # sequence_scores.fill_(0.0)

        # Initialize the input vector
        y_prev = torch.zeros((batch_size * beam_width), device = self.device).fill_(self.num_classes)

        # Store decisions for backtracking
        stored_scores          = list()
        stored_predecessors    = list()
        stored_emitted_symbols = list()

        for i in range(self.max_len_labels):
            output, state = decoder(inflated_encoder_feats, state, y_prev)
            log_softmax_output = F.log_softmax(output, dim=1)

            sequence_scores = _inflate(sequence_scores, self.num_classes, 1)
            sequence_scores += log_softmax_output
            scores, candidates = sequence_scores.view(batch_size, -1).topk(beam_width, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            y_prev = (candidates % self.num_classes).view(batch_size * beam_width)
            sequence_scores = scores.view(batch_size * beam_width, 1)

            # Update fields for next timestep
            predecessors = (candidates / self.num_classes + pos_index.expand_as(candidates)
                           ).view(batch_size * beam_width, 1)
            state = state.index_select(1, predecessors.squeeze())

            # Update sequence socres and erase scores for <eos> symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = y_prev.view(-1, 1).eq(eos)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(y_prev)

        # Do backtracking to return the optimal values
        #====== backtrak ======#
        # Initialize return variables given different types
        p = list()
        l = [[self.max_len_labels] * beam_width for _ in range(batch_size)]  # Placeholder for lengths of top-k sequences

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = stored_scores[-1].view(batch_size, beam_width).topk(beam_width)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * batch_size  # the number of EOS found
                                            # in the backward loop below for each batch
        t = self.max_len_labels - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (sorted_idx + pos_index.expand_as(sorted_idx)).view(batch_size * beam_width)
        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = stored_emitted_symbols[t].index_select(0, t_predecessors)
            t_predecessors = stored_predecessors[t].index_select(0, t_predecessors).squeeze()
            eos_indices = stored_emitted_symbols[t].eq(eos).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / beam_width)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * beam_width + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = stored_predecessors[t][idx[0]]
                    current_symbol[res_idx] = stored_emitted_symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = stored_scores[t][idx[0], [0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(beam_width)
        for b_idx in range(batch_size):
            l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + pos_index.expand_as(re_sorted_idx)).view(batch_size*beam_width)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        p = [step.index_select(0, re_sorted_idx).view(batch_size, beam_width, -1) for step in reversed(p)]
        p = torch.cat(p, -1)[:,0,:]
        return p, s[:,0] #p[batch_size, beam_width, max_len_labels], s[batch_size, beam_width]


class AttentionUnit(nn.Module):
    def __init__(self, sDim, xDim, attDim):
        super(AttentionUnit, self).__init__()

        self.sDim = sDim
        self.xDim = xDim
        self.attDim = attDim

        self.sEmbed = nn.Linear(sDim, attDim)
        self.xEmbed = nn.Linear(xDim, attDim)
        self.wEmbed = nn.Linear(attDim, 1)

    def forward(self, x, sPrev):
        '''
        x shape [[batch_size, length_of_labels, xDim]]
        '''
        batch_size, T, _ = x.size()                       # [batch_size x T x xDim]
        x = x.view(-1, self.xDim)                         # [(batch_size x T) x xDim]
        xProj = self.xEmbed(x)                            # [(batch_size x T) x attDim]
        xProj = xProj.view(batch_size, T, -1)             # [batch_size x T x attDim]

        sPrev = sPrev.squeeze(0)
        sProj = self.sEmbed(sPrev)                        # [batch_size x attDim]
        sProj = torch.unsqueeze(sProj, 1)                 # [batch_size x 1 x attDim]
        sProj = sProj.expand(batch_size, T, self.attDim)  # [batch_size x T x attDim]

        sumTanh = torch.tanh(sProj + xProj)
        sumTanh = sumTanh.view(-1, self.attDim)

        vProj = self.wEmbed(sumTanh) # [(batch_size x T) x 1]
        vProj = vProj.view(batch_size, T)

        alpha = F.softmax(vProj, dim=1) # attention weights for each sample in the minibatch

        return alpha

class DecoderUnit(nn.Module):
    def __init__(self, sDim, xDim, yDim, attDim):
        super(DecoderUnit, self).__init__()
        self.sDim = sDim
        self.xDim = xDim
        self.yDim = yDim
        self.attDim = attDim
        self.emdDim = attDim

        self.attention_unit = AttentionUnit(sDim, xDim, attDim)
        self.tgt_embedding = nn.Embedding(yDim+1, self.emdDim) # the last is used for <BOS> 
        self.gru = nn.GRU(
            input_size=xDim+self.emdDim, 
            hidden_size=sDim, 
            batch_first=True
        )
        self.fc = nn.Linear(sDim, yDim)

    def forward(self, x, sPrev, yPrev):
        # x: feature sequence from the image decoder.
        batch_size, T, _ = x.size()
        alpha = self.attention_unit(x, sPrev)
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
        yProj = self.tgt_embedding(yPrev.long())
        
        self.gru.flatten_parameters()
        output, state = self.gru(torch.cat([yProj, context], 1).unsqueeze(1), sPrev)
        output = output.squeeze(1)

        output = self.fc(output)
        return output, state