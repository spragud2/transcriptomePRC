# transcriptomePRC
Quickly search a large database of sequences for regions that may associate with Polycomb Repressive Complexes. Program scores transcripts and saves locations of significant regions.<br/>

This script is optimized for parallelization and for very large searches <br/>

## Usage example
`Ex. python qSEEKRscanr.py -t fasta.fa -k 4 -n 6 -w 850 -s 85`
<br/>

## Parameter Definitions
-t path to fasta set of sequences <br/>
-k value of k for kmer counting (default = 5) <br/>
--thresh percentile threshold to count a sequence regions as a 'hit' to a query (default = 95) <br/>
-n number of CPU cores desired (default = multiprocessing.cpu_count() - 1 ) <br/>
-w windowing size to tile sequences (default = 1000) <br/>
-s how many bp to slide windows (default = 100) <br/>
 <br/>
 
### Other requirements (see folders in repo as example)
Requires:
1. Set of query sequences that have some known function or property <br/>
2. Pre-calculated mean and std for k-mers in reference set <br/>
3. Pre-calculated distribution of pearon's correlations between query and reference to calculate rank <br/>

 <br/>
