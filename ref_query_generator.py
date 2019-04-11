from seekr.kmer_counts import BasicCounter
import numpy as np

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

for k in [4,5,6]:
    ref = BasicCounter(infasta='../../standard_sequences/gencode.vM17.lncRNA_transcripts.fa',k=k,mean=False,std=False,log2=False)
    ref.get_counts()
    ref_mean,ref_std = np.mean(ref.counts,axis=0),np.std(ref.counts,axis=0)
    ref.counts = (ref.counts - ref_mean)/ref_std
    ref.counts+= abs(ref.counts.min()) + 1
    ref.counts = np.log2(ref.counts)
    queries = BasicCounter(infasta='./queries/queries.fa',k=k,mean=False,std=False,log2=False)
    queries.get_counts()
    queries.counts = (queries.counts - ref_mean)/ref_std
    queries.counts+= abs(queries.counts.min()) + 1
    queries.counts = np.log2(queries.counts)
    qSEEKRmat = rectCorr(queries.counts,ref.counts)

    np.save(f'./{k}ref.npy',qSEEKRmat)
    np.save(f'./{k}mean.npy',ref_mean)
    np.save(f'./{k}std.npy',ref_std)
