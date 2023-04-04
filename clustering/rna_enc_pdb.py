"""Encoder/Decoder for RNA sequences."""

import itertools
import re

class RNAEncoder():
    
    def __init__(self, L=None, args=None):
        
        self._L = L
        self._args = args
        
        if self._args == "seq":
            self.BASES = ['G', 'C', 'A', 'U', 'N']
        else:
            self.BASES = self._get_pairings_list()
        
        tokens = self._tokens()
        ids = range(len(tokens))
        self._ids_to_tokens = dict(zip(ids, tokens))
        self._tokens_to_ids = dict(zip(tokens,ids))
        
        
    @property
    def vocab_size(self):
        return len(self._ids_to_tokens)
    
    
    def _get_pairings_list(self):
        pairs_lst = []
        pairs = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']
        for pair in pairs:
            lst = [(pair, i) for i in range(self._L)]
            pairs_lst.extend(lst)
        return pairs_lst
    
    
    def _tokens(self):
        return list(itertools.product(self.BASES, repeat=1))
        
        
    
    def encode(self, s):
        
        if self._args == "seq":
            bases = re.findall(r'[ACGUN]', s)
            
        elif self._args == "struc":
            
            tuples_lst = []
            for i in s:
                if i[1] == i[3]:
                    B = tuple(('NN',i[2]))
                    tuples_lst.append(B)
                else:
                    B = tuple((i[1] + i[3], i[2]))
                    tuples_lst.append(B)

            bases = tuples_lst

            
        ids = [self._tokens_to_ids[(i,)] for i in bases]
        
        if len(bases) < self._L:
            pad = [-1] * (self._L - len(bases))
            padded_seq = ids + pad
            ids = padded_seq
        
        return ids
    
    
    def decode(self, ids):
        
        bases = []
        for i in ids:
            if i == -1:
                bases.extend(["<PAD>"])
            else:
                bases.extend(self._ids_to_tokens[i])
            
        return bases
            
        
    