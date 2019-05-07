import pandas as pd
import numpy as np
from seekr.fasta_reader import Reader
from itertools import product
from collections import defaultdict
from scipy.stats import pearsonr
from seekr.kmer_counts import BasicCounter
import glob
from os.path import basename
import pickle
import argparse
import multiprocessing
from tqdm import tqdm
'''
# This version of the script parallelizes computation
# And completely vectorizes the calculation of the correlation matrix
'''
###########################################################################

###########################################################################

#
# def rectCorr(queries_kmers,ref_kmers):
#     queries_kmers = (queries_kmers.T - np.mean(queries_kmers, axis=1)).T
#     ref_kmers = (ref_kmers.T - np.mean(ref_kmers, axis=1)).T
#     #Transpose tile matrix and dot product queries against tiles
#     cov = queries_kmers.dot(ref_kmers.T)
#     #Get the euclidean norm of each query and tile
#     qNorm = np.linalg.norm(queries_kmers, axis=1)
#     tNorm = np.linalg.norm(ref_kmers, axis=1)
#     #Outer vector multiplication to match query to tiles
#     norm = np.outer(qNorm, tNorm)
#     #Correlation matrix calculation
#     qSEEKRmat = cov/norm
#     return qSEEKRmat.T
def classify(seq, k, lrTab):
    """ Classify seq using given log-ratio table.  We're ignoring the
        initial probability for simplicity. """
    seq = seq.upper()
    bits = 0
    nucmap = { 'A':0, 'T':1, 'C':2, 'G':3 }
    rowmap = dict(zip([''.join(p) for p in product(['A','T','C','G'],repeat=k-1)],range(4**(k-1))))
    for kmer in [seq[i:i+k] for i in range(len(seq)-k+1)]:
        if 'N' not in kmer:
            i, j = rowmap[kmer[:k-1]], nucmap[kmer[-1]]
            bits += lrTab[i, j]
    return bits

def qSEEKR(k,ae4Tbl,bTbl, target, w, s):
    t_h, t_s = target
    window, slide = w, s
    hits = {}
    tiles = [t_s[i:i+window] for i in range(0, len(t_s), slide)]

    bSeq = np.array([classify(tile,k,bTbl) for tile in tiles])
    ae4Seq = np.array([classify(tile,k,ae4Tbl) for tile in tiles])
    qSEEKRmat = np.column_stack((bSeq,ae4Seq))
    hits_idx = np.argwhere(qSEEKRmat > 1)
    tot_scores = np.sum(qSEEKRmat > 1,axis=0) / len(t_s)
    tot_scores = np.sum(tot_scores)
    return t_h, [hits_idx, tot_scores]
###########################################################################

###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-t",type=str,help='Path to target fasta file')
parser.add_argument('-k', type=int,default=5)
parser.add_argument('--thresh', type=int,
                    help='Percentile to select hits', default=99)
parser.add_argument('-n', type=int, help='Number of processors,default = number cpus avail',
                    default=multiprocessing.cpu_count()-1)
parser.add_argument('-w', type=int, help='Window for tile size', default=1000)
parser.add_argument(
    '-s', type=int, help='How many bp to slide tiles', default=100)
args = parser.parse_args()

###########################################################################

###########################################################################
#Path to known functional domains

target_path = args.t
target_head, target_seq = Reader(
    target_path).get_headers(), Reader(target_path).get_seqs()
target_dict = dict(zip(target_head, target_seq))

b_model = np.load('./bmodel_2mers.mkv.npy')
ae4_model = np.load('./AE4model_2mers.mkv.npy')
lncome_model = np.load('./genomemodel_2mers.mkv.npy')
###########################################################################

###########################################################################
bTbl = np.log2(b_model) - np.log2(lncome_model)
ae4Tbl = np.log2(ae4_model) - np.log2(lncome_model)
###########################################################################
'''
Parallelize transcript computations
'''
###########################################################################
with multiprocessing.Pool(args.n) as pool:
    ha = pool.starmap(qSEEKR, product(
        *[[args.k], [ae4Tbl], [bTbl],list(target_dict.items()), [args.w], [args.s]]))
    hits = dict(ha)
pickle.dump(hits, open(f'../{basename(args.t)[:-3]}_{args.k}_{args.thresh}_{args.w}win_{args.s}slide_scores_MARKOV.p', 'wb'))
