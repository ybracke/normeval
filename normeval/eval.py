import Levenshtein
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def seqsdist(seq1, seq2, dist_metric):
    ''' 
    String distances of elements in 2 lists at same index position
    
    `seq1`, `seq2` are lists of strings, must have same length
    dist_metric can currently only be 'lev' for Levenshtein.distance
    '''
    
    # Check 
    if not( len(seq1) == len(seq2)):
        raise ValueError(
            f'Sequence length must match: {len(seq1)}!={len(seq2)}')

    # Lookup metric method
    metrics = {'lev' : Levenshtein.distance} # TODO add more
    dist_metric = metrics[dist_metric]

    n = len(seq1)

    # array with string distances 
    try: 
        dists = [dist_metric(seq1[i],seq2[i]) for i in range(n)]
    except:
        raise
    return dists



class Evaluator():

    def __init__(self, pred=[], gold=[]):
        ''' 
        Evaluator object for computing similarity between two word lists

        Use cases: Comparing the output of automatic normalization with
        a gold normalization,
        
        Initialize `Evaluator` empty or with two word lists 
        '''

        self.pred = pred
        self.gold = gold
        self.n_corpus = len(gold)

    def _get_and_enter_stats(self, array, results, key):
        '''
        Parameters
        ----------

        array : list/array of numbers, some string metric
        results : dict to be updated
        key : entry to make in `results`, typically name of string metric
        '''
        results[key] = {}
        results[key]['mean'] = np.mean(array)
        results[key]['std'] = np.std(array)
        results[key]['max'] = max(array)
        results[key]['min'] = min(array)
        results[key]['median'] = np.median(array)
        results[key]['q1,q2,q3,q4'] = list(
            np.quantile(array, [.25,.5,.75,1]))

    def evaluate(self, methods):
        ''' 
        Perform one or more evaluation methods on data and return results 

        `methods` can be either `'all'` or a list of strings with the 
        following possible elements: `'lev', 'levn', 'acc', 'macro-f1'`
        '''

        results = {}

        # Levenshtein distance (LD)
        if ('lev' in methods) or (methods == 'all'):
            
            # Vanilla LD
            dists = seqsdist(self.gold, self.pred, dist_metric='lev')
            self._get_and_enter_stats(dists, results, "Levenshtein")

            # LD normalized by length of longer string
            if ('levn' in methods) or (methods == 'all'): 
                dists_norm = [dists[i] / len(max(self.gold[i], self.pred[i])) 
                              for i in range(self.n_corpus)]
                self._get_and_enter_stats(dists_norm, results, 
                                          "Levenshtein_norm")

        # Accuracy
        if ('acc' in methods) or (methods == 'all'):
            results['accuracy'] = accuracy_score(self.gold, self.pred)

        # Macro-F1
        if ('macro-f1' in methods) or (methods == 'all'):
            results['macro-F1'] = f1_score(self.gold, self.pred,
                                           average='macro', zero_division=0)

        return results


