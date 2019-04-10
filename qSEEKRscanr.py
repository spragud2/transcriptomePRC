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
###########################################################################

###########################################################################
def count_kmers(fa,k,k_map):
    counts = []
    for seq in fa:
        currlen = len(seq)
        vals = np.zeros(len(k_map))
        for i in range(currlen-k+1):
            if seq[i:i+k] in k_map:
                vals[k_map[seq[i:i+k]]]+=1
        vals = 1000*(vals/currlen)
        counts.append(vals)
    return np.array(counts)

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


def qSEEKR(refs, k, Q, target, w, s,mean,std,k_map):
    t_h, t_s = target
    window, slide = w, s
    hits = {}
    tiles = [t_s[i:i+window] for i in range(0, len(t_s), slide)]
    threeprime_hang = len(t_s) % slide
    if threeprime_hang != 0:
       tiles[-1] = tiles[-1]+t_s[-threeprime_hang:]

    tCounts = count_kmers(tiles,k,k_map)
    tCounts = (tCounts - mean)/std
    tCounts = np.log2(tCounts + np.abs(np.min(tCounts))+1)
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
parser.add_argument('-k', type=int,default=5)
parser.add_argument('--thresh', type=int,
                    help='Percentile to select hits', default=99)
parser.add_argument('-n', type=int, help='Number of processors,default = number cpus avail',
                    default=multiprocessing.cpu_count()-1)
parser.add_argument('-w', type=int, help='Window for tile size', default=1000)
parser.add_argument(
    '-s', type=int, help='How many bp to slide tiles', default=100)
args = parser.parse_args()

kmers = [''.join(p) for p in product('AGTC',repeat=args.k)]
kmer_map = dict(zip(kmers,range(0,4**args.k)))
###########################################################################

###########################################################################
#Path to known functional domains
query_path = './queries/queries.fa'
target_path = args.t
target_head, target_seq = Reader(
    target_path).get_headers(), Reader(target_path).get_seqs()
target_dict = dict(zip(target_head, target_seq))

queries = dict(zip(Reader(query_path).get_headers(),
                   Reader(query_path).get_seqs()))
###########################################################################

###########################################################################
mean_paths = [f for f in glob.iglob('./stats/*mean.npy')]
std_paths = [f for f in glob.iglob('./stats/*std.npy')]

means = {}
for mean_path in mean_paths:
    means[basename(mean_path)] = np.load(mean_path)

stds = {}
for std_path in std_paths:
    stds[basename(std_path)] = np.load(std_path)
mean = means[f'{args.k}mean.npy']
std = stds[f'{args.k}std.npy']
###########################################################################

###########################################################################
ref = np.load(f'./refs/{args.k}ref.npy')
###########################################################################

###########################################################################

queryseqs = list(queries.values())
query_counts = count_kmers(queryseqs,args.k,kmer_map)
query_counts = (query_counts - mean)/std
query_counts = np.log2(query_counts + np.abs(np.min(query_counts))+1)
print(query_counts)
1/0

Q = query_counts
# querymap = dict(zip(range(len(queries)), list(queries.keys())))

# query_percentile = {}
# for i in range(len(queries)):
#     percentile = np.percentile(refs_new[querymap[i]], args.thresh)
#     query_percentile[querymap[i]] = percentile
#
# q_arr = np.array(list(query_percentile.values()))
#This performs standard SEEKR for the query

percentiles = np.percentile(ref, args.thresh, axis=0)
###########################################################################
'''
Parallelize transcript computations
'''
###########################################################################
with multiprocessing.Pool(args.n) as pool:
    ha = pool.starmap(qSEEKR, product(
        *[[percentiles], [args.k], [Q], list(target_dict.items()), [args.w], [args.s],[mean],[std],[kmer_map]]))
    hits = dict(ha)
pickle.dump(hits, open(f'../{basename(args.t)[:-3]}_{args.k}_scores_test.p', 'wb'))
#1/0
