import pickle
import os
import collections
import sys

# sys.path.append('./pycocoevalcap')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
#from pycocoevalcap.cider.cider import Cider

import torch
import torch.nn as nn

import onmt
import pdb
count_n = 0

class Evaluate(nn.Module):
    def __init__(self, vocab_file='graph2text/data/vocabs.txt'):
        super(Evaluate, self).__init__()
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Rouge(), "ROUGE_L")
            ]
        with open(vocab_file, encoding='utf-8') as f:
            vocab_list = f.readlines()
        self.vocab = [_.strip('\n') for _ in vocab_list]
        self.padding_idx = self.vocab.index('<blank>')

    def forward(self, output1, target):

        output1_p, output1_idx = output1.max(dim=-1)
        rewards = []
        for i in range(target.size(0)):
            # action, select_log_p, entropy = self.impl(torch.softmax(output2[:, i], dim=-1))
            target_sentence = [self.vocab[x.item()] for x in target[i]]
            output1_sentence = [self.vocab[x.item()] for x in output1_idx[i]]
            
            target_sentence = {'sentence': [' '.join(target_sentence)]}
            output1_sentence = {'sentence': ' '.join(output1_sentence)}

            bleu_1 = self.evaluate(live=True, cand=output1_sentence, ref=target_sentence)
            rewards.append(bleu_1 / 100.0)
        return torch.tensor(rewards)

    def stats(self, loss, pred, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        # pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def impl(self, probs):
        m = torch.distributions.Categorical(probs=probs)
        action = m.sample().view(-1)
        select_log_p = m.log_prob(action)
        entropy = m.entropy()
        return action, select_log_p, entropy

    def convert(self, data):
        if isinstance(data, basestring):
            return data.encode('utf-8')
        elif isinstance(data, collections.Mapping):
            return dict(map(convert, data.items()))
        elif isinstance(data, collections.Iterable):
            return type(data)(map(convert, data))
        else:
            return data

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores

    def evaluate(self, live=False, **kwargs):
        if live:
            temp_ref = kwargs.pop('ref', {})
            cand = kwargs.pop('cand', {})
        else:
            reference_path = kwargs.pop('ref', '')
            candidate_path = kwargs.pop('cand', '')

            # load caption data
            with open(reference_path, 'rb') as f:
                temp_ref = pickle.load(f)
            with open(candidate_path, 'rb') as f:
                cand = pickle.load(f)

        # make dictionary
        hypo = {}
        ref = {}
        i = 0
        for vid, caption in cand.items():
            hypo[i] = [caption]
            ref[i] = temp_ref[vid]
            i += 1

        # compute scores
        final_scores = self.score(ref, hypo)
        final_scores = final_scores['Bleu_4']  # 0.1 * final_scores['Bleu_1'] + 0.2 * final_scores['Bleu_2'] + \
                #    0.3 * final_scores['Bleu_3'] + 0.4 * final_scores['Bleu_4']
        return final_scores


if __name__ == '__main__':
    '''
    cands = {'generated_description1': 'how are you', 'generated_description2': 'Hello how are you'}
    refs = {'generated_description1': ['what are you', 'where are you'],
           'generated_description2': ['Hello how are you', 'Hello how is your day']}
    '''
    cands = ['how are you', 'Hello how are you']
    refs = ['how are you', 'Hello how are you']
    cands = {'generated_description'+str(i):x.strip() for i,x in enumerate(cands)}
    refs = {'generated_description'+str(i):[x.strip()] for i,x in enumerate(refs)}
    x = Evaluate()
    final_scores = x.evaluate(live=True, cand=cands, ref=refs)
    print(final_scores)
