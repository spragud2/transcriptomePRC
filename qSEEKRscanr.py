from scipy.stats import percentileofscore
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
'''
# This version of the script parallelizes computation
# And completely vectorizes the calculation of the correlation matrix
'''
def rectCorr(queries_kmers,ref_kmers):
    queries_kmers = (queries_kmers.T - np.mean(queries_kmers, axis=1)).T
    ref_kmers = (ref_kmers.T - np.mean(ref_kmers, axis=1)).T
    #Transpose tile matrix and dot product queries against tiles
    cov = queries_kmers.dot(ref_kmers.T)
    #Get the euclidean norm of each query and tile
    qNorm = np.linalg.norm(queries_kmers, axis=1)
    tNorm = np.linalg.norm(ref_kmers, axis=1)
    #Outer vector multiplication to match query to tiles
    norm = np.outer(qNorm, tNorm)
    #Correlation matrix calculation
    qSEEKRmat = cov/norm
    return qSEEKRmat.T


def qSEEKR(refs, k, Q, target, w, s):
    t_h, t_s = target
    window, slide = w, s
    hits = {}
    tiles = [t_s[i:i+window] for i in range(0, len(t_s), slide)]
    threeprime_hang = len(t_s) % slide
    if threeprime_hang != 0:
       tiles[-1] = tiles[-1]+t_s[-threeprime_hang:]

    tile_counter = BasicCounter(k=args.k, mean=mean,  std=std, silent=True,log2=False)
    tile_counter.seqs = tiles
    tile_counter.get_counts()
    tCounts = tile_counter.counts

    #Completely vectorized implementation of the old 'dSEEKR'
    #Convert row means in matrices to 0
    qSEEKRmat = rectCorr(Q,tCounts)
    hits_idx = np.argwhere(qSEEKRmat > refs)
    tot_scores = np.sum(qSEEKRmat > refs) / len(t_s)
    return t_h, [qSEEKRmat, hits_idx, tot_scores]
###########################################################################

###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-t")
parser.add_argument('-k', type=int)
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
query_path = './queries/queries.fa'
target_path = args.t
target_head, target_seq = Reader(
    target_path).get_headers(), Reader(target_path).get_seqs()
queries = dict(zip(Reader(query_path).get_headers(),
                   Reader(query_path).get_seqs()))
###########################################################################

###########################################################################
mean_paths = [f for f in glob.iglob('./stats/*mean.npy')]
means = {}
for mean_path in mean_paths:
    means[basename(mean_path)] = np.load(mean_path)
std_paths = [f for f in glob.iglob('./stats/*std.npy')]

stds = {}
for std_path in std_paths:
    stds[basename(std_path)] = np.load(std_path)
mean = means[f'{args.k}mean.npy']
std = stds[f'{args.k}std.npy']
###########################################################################

###########################################################################
target_dict = dict(zip(target_head, target_seq))
refs = {}
ref_paths = [f for f in glob.iglob(f'./refs/*{args.k}_ref.npy')]
for ref_path in ref_paths:
    curr_npy = np.load(ref_path)
    refs[basename(ref_path)] = curr_npy

refs_keys = [f'>{i[:-10]}' for i in refs.keys()]
refs_new = {}
for i, k in enumerate(refs):
    refs_new[refs_keys[i]] = refs[k]
###########################################################################

###########################################################################

query_counter = BasicCounter(k=args.k, mean=mean, std=std, silent=True,log2=False)
query_counter.seqs = list(queries.values())
query_counter.get_counts()
query_counts = query_counter.counts

queries = dict(zip(queries.keys(), query_counts))
Q = np.array([list(i) for i in queries.values()])

querymap = dict(zip(range(len(queries)), list(queries.keys())))

query_percentile = {}
for i in range(len(queries)):
    percentile = np.percentile(refs_new[querymap[i]], args.thresh)
    query_percentile[querymap[i]] = percentile

q_arr = np.array(list(query_percentile.values()))
#This performs standard SEEKR for the query

refs_arr = np.array([list(i) for i in refs.values()])
percentiles = np.percentile(refs_arr, args.thresh, axis=1)
###########################################################################
'''
Parallelize transcript computations
'''
###########################################################################
with multiprocessing.Pool(args.n) as pool:
    ha = pool.starmap(qSEEKR, product(
        *[[q_arr], [args.k], [Q], list(target_dict.items()), [args.w], [args.s]]))
    hits = dict(ha)
pickle.dump(hits, open(f'../{basename(args.t)[:-3]}_{args.k}_scores.p', 'wb'))
#1/0
