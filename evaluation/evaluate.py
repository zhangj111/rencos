from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from meteor.meteor import Meteor
import numpy as np
import sys


def main(hyp, ref, len):
    with open(hyp, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split()[:len])] for k, v in enumerate(hypothesis)}
    with open(ref, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}

    score_Bleu, scores_Bleu = Bleu(4).compute_score(gts, res)
    print("Bleu_1: "), np.mean(scores_Bleu[0])
    print("Bleu_2: "), np.mean(scores_Bleu[1])
    print("Bleu_3: "), np.mean(scores_Bleu[2])
    print("Bleu_4: "), np.mean(scores_Bleu[3])

    score_Meteor, scores_Meteor = Meteor().compute_score(gts, res)
    print("Meteor: "), score_Meteor

    score_Rouge, scores_Rouge = Rouge().compute_score(gts, res)
    print("ROUGe: "), score_Rouge

    score_Cider, scores_Cider = Cider().compute_score(gts, res)
    print("Cider: "), score_Cider

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], eval(sys.argv[3]))