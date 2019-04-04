# transcriptomePRC
Quickly search a large database of sequences for regions that may associate with Polycomb Repressive Complexes. Program scores transcripts and saves locations of significant regions.

-t path to fasta set of sequences
-k value of k for kmer counting (default = 5)
-n number of CPU cores desired (default = multiprocessing.cpu_count() - 1 )
-w windowing size to tile sequences (default = 1000)
-s how many bp to slide windows (default = 100)

Requires:
1. Set of query sequences that have some known function or property
2. Pre-calculated mean and std for k-mers in reference set
3. Pre-calculated distribution of pearon's correlations between query and reference to calculate rank


Ex. python -t fasta.fa -k 4 -n 6 -w 850 -s 85
