from seekr.kmer_counts import BasicCounter
import numpy as np

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
    qSEEKRmat = SEEKRrectCorr(queries.counts,ref.counts)

    np.save(f'./{k}ref.npy',qSEEKRmat)
    np.save(f'./{k}mean.npy',ref_mean)
    np.save(f'./{k}std.npy',ref_std)
