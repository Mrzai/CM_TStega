from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ciderD_scorer import CiderScorer
import pdb

class CiderD:
    """
    Main Class to compute the CIDEr metric
    """
    def __init__(self, n=4, sigma=6.0, df="corpus"):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set which where to compute document frequencies from
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        for res_id in res:

            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            tmp_cider_scorer += (hypo[0], ref)

        (score, scores) = tmp_cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr-D"
