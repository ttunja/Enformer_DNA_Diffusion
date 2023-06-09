#!/usr/bin/env python3
import os
import urllib.request
import pandas as pd
import gzip
import shutil

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Download data
urllib.request.urlretrieve('http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz', 'data/hg38.fa.gz')
with gzip.open('data/hg38.fa.gz', 'rb') as f_in:
    with open('data/genome.fa', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
urllib.request.urlretrieve('https://www.dropbox.com/s/n549ucfqu0u9tu6/master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt?dl=1', 'data/MASTER_DNA_DIFFUSION_ALL_SEQS.txt')
urllib.request.urlretrieve('https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz', 'data/clinvar.vcf.gz')
urllib.request.urlretrieve('https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes', 'data/hg38.chrom.sizes')
urllib.request.urlretrieve('https://www.dropbox.com/s/a9ggrhn3626x0di/DNA_DIFFUSION_ALL_SEQS.txt?dl=1', 'data/DNA_DIFFUSION_ALL_SEQS.txt')
urllib.request.urlretrieve('https://www.encodeproject.org/files/ENCFF413AHU/@@download/ENCFF413AHU.bigWig', 'data/ENCFF413AHU.bigWig')
urllib.request.urlretrieve('https://www.encodeproject.org/files/ENCFF093VXI/@@download/ENCFF093VXI.bigWig', 'data/ENCFF093VXI.bigWig')
urllib.request.urlretrieve('https://github.com/pinellolab/DNA-Diffusion/raw/main/src/dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt', 'data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt')
urllib.request.urlretrieve('https://www.dropbox.com/s/oqpn784x34f6pcq/random_regions_train_generated_genome_10k.txt?dl=1', 'data/random_regions_train_generated_genome_10k.txt')
urllib.request.urlretrieve('https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz', 'data/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz')

# Print contents of data directory
print(os.listdir('data'))
