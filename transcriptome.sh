#!/bin/bash

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=2g
#SBATCH -t 5-00:00:00

for f in *.fa; do sbatch -J $f -N 1 -n 24 -t 50:00:00 --wrap="python qSEEKRscanr.py  -t $f -k 2 -w 100 -s 20"; done
