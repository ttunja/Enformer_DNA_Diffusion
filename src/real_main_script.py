
import pandas as pd
from IPython.display import HTML, display
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pyBigWig
import igv_notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os 
import pickle as pkl
import wget 
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from enformer import Enformer, FastaStringExtractor
from enformerops import EnformerOps, SEQ_EXTRACT


model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = 'data/genome.fa'
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
df_targets[df_targets.apply(lambda x : '' in x['description'] and  'CAGE' in x['description'] , 1)]
df_sizes = pd.read_table('data/hg38.chrom.sizes', header=None).head(22)

# Enformer TensorFlow base code
SEQUENCE_LENGTH = 393216

model = Enformer(model_path)

fasta_extractor = FastaStringExtractor(fasta_file)


def main():
    # !wget https://www.dropbox.com/s/n549ucfqu0u9tu6/master_random_frezzed_regions_train_test_validation_generated_genome_all_dataset.txt?dl=1  -O  MASTER_DNA_DIFFUSION_ALL_SEQS.txt
    SEQS = SEQ_EXTRACT('data/MASTER_DNA_DIFFUSION_ALL_SEQS.txt')
    SEQS
    # -Add the random for our full dataset.
    # -Dont remove hesct0 (use h1esc) use NA for cage
    # -Generate 10k random sequences (remove regions with NN)
    # -Add the CAGE tss(1kb+/-) prediction for the other 3 celltypes on the 440k.
    # -Dont show tss in the final sequences.

    # eops.add_track({
    #     'name': 'DNAse:hESC_enformer',
    #     'file': None,
    #     'color': 'BLUE',
    #     'type': 'enformer',
    #     'id': 123,
    #     'log': False
    # })

    def load_tracks_eops(eops):
        
        eops.add_track({
            'name': 'DNASE:GM12878_enformer',
            'file': None,
            'color': 'BLUE',
            'type': 'enformer',
            'id': 12,
            'log': False
        })

        eops.add_track({
            'name': 'DNASE:K562_enformer',
            'file': None,
            'color': 'BLUE',
            'type': 'enformer',
            'id': 121,
            'log': False
        })

        eops.add_track({
            'name': 'DNASE:HepG2_enformer',
            'file': None,
            'color': 'BLUE',
            'type': 'enformer',
            'id': 27,
            'log': False
        })
        
        eops.add_track({
            'name': 'DNASE:H1esc_enformer',
            'file': None,
            'color': 'BLUE',
            'type': 'enformer',
            'id': 19,
            'log': False
        })

        eops.add_track({
            'name': 'CAGE:K562_enformer',
            'file': None,
            'color': 'RED',
            'type': 'enformer',
            'id': 5111,
            'log': True
        })

        eops.add_track({
            'name': 'CAGE:GM12878_enformer',
            'file': None,
            'color': 'RED',
            'type': 'enformer',
            'id': 5110,
            'log': True
        })

        eops.add_track({
            'name': 'CAGE:HepG2_enformer',
            'file': None,
            'color': 'RED',
            'type': 'enformer',
            'id': 5109,
            'log': True
        })
    DEMO=True
    all_data = SEQS.data.copy()
    MODIFY_PREFIX= '1_'
    #run lucas_test
    #papermill  5_31_2023_enformer_CAGE_DNAse_bias_experiment.ipynb  demo.ipynb
    ## Random sequences
    print('Remove all head 3')
    eops = EnformerOps()
    eops.load_data(SEQS.extract_seq('GENERATED', 'GM12878')['SEQUENCE'].values.tolist())
    load_tracks_eops(eops) #loading the tracks

    ran_seqs = all_data[all_data['TAG'] == 'RANDOM_GENOME_REGIONS'].copy()
    print (ran_seqs['TAG'].unique(), ran_seqs.shape)

    subset_ran_seqs = ran_seqs[['chrom', 'start', 'end', 'ID']]

    ran_seq_list = subset_ran_seqs.values.tolist() # removing demo
    if DEMO:
        ran_seq_list=ran_seq_list[:3]

    captured_values = []
    for s in ran_seq_list: 
        try:
            s_in = [s[0], int(s[1]), int(s[2])]
            id_seq = s[3] 
            list_bw = eops.generate_plot_number(model, 0, interval_list=s_in, wildtype=True, 
                                            show_track=False, modify_prefix=MODIFY_PREFIX,
                                            fasta_file=fasta_file, sequence_length=SEQUENCE_LENGTH) 
        except RuntimeError:
        # Infrequent "the entries are out of order" error for some random seqs 
            continue
        try:
            out_in = eops.extract_from_position(s_in, as_dataframe=True)
            out_in = out_in.mean()
            out_in['SEQ_ID'] = id_seq
            out_in['TARGET_NAME'] = 'ITSELF'
            columns_names = out_in.copy()
            captured_values.append( out_in  )
        except ValueError:
        # Infrequent "All arrays must be of the same length" error
            continue

    df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

    df_out.to_csv('DNASE_RANDOM_SEQS.TXT', sep='\t', index=None)
    ## Promoters
    # ['RANDOM_GENOME_REGIONS']        (5000)    (MEASURING: 200bp)       #DNASE (K562,HEPG2,GM, H1ESC) # CAGE (K562,HEPG2,GM)
    # ['PROMOTERS']                    (20013)  (MEASURING: 2000bp)       #DNASE (K562,HEPG2,GM, H1ESC) # CAGE (K562,HEPG2,GM)
    # ['training' 'test' 'validation'] (47872)   (MEASURING: 200bp)       #DNASE (K562,HEPG2,GM, H1ESC) # CAGE (K562,HEPG2,GM)

    # #2X (ENHANCER AND GATA1)  #DNASE (K562,HEPG2,GM, H1ESC) # CAGE (K562,HEPG2,GM)
    # ['GENERATED']                    (400000)   (MEASURING: ENH(200bp) GATA1_GB( 7925bp) ) 
                                

    print('Remove all head 3')
    eops = EnformerOps()
    eops.load_data(SEQS.extract_seq('GENERATED', 'GM12878')['SEQUENCE'].values.tolist())
    load_tracks_eops(eops) #loading the tracks

    ran_seqs = all_data[all_data['TAG'] == 'PROMOTERS'].copy()
    print (ran_seqs['TAG'].unique(), ran_seqs.shape)

    subset_ran_seqs = ran_seqs[['chrom', 'start', 'end', 'ID']]
    #ran_seq_list = subset_ran_seqs.head(3).values.tolist()
    ran_seq_list = subset_ran_seqs.values.tolist()
    if DEMO:
        ran_seq_list=ran_seq_list[:3]

    captured_values = []
    for s in tqdm(ran_seq_list): 
        #print(s)
        try:
            s_in = [s[0], int(s[1]), int(s[2])]
            id_seq = s[3] 
            list_bw = eops.generate_plot_number(model, 0, interval_list=s_in, wildtype=True, 
                                            show_track=False, modify_prefix=MODIFY_PREFIX) 
        except RuntimeError:
        # Infrequent "the entries are out of order" error for some random seqs 
            continue
        try:
            out_in = eops.extract_from_position(s_in, as_dataframe=True)
            #print (out_in.shape)
            out_in = out_in.mean()
            out_in['SEQ_ID'] = id_seq
            out_in['TARGET_NAME'] = 'ITSELF'
            columns_names = out_in.copy()
            captured_values.append(out_in)
        except ValueError:
        # Infrequent "All arrays must be of the same length" error
            continue

    df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

    df_out.to_csv('PROMOTERS_SEQS.TXT', sep='\t', index=None)
    ## TRAINING
    print('Remove all head 3')
    eops = EnformerOps()
    eops.load_data(SEQS.extract_seq('GENERATED', 'GM12878')['SEQUENCE'].values.tolist())
    load_tracks_eops(eops) #loading the tracks

    ran_seqs = all_data[all_data['TAG'].apply(lambda x : x in {'training', 'test', 'validation'})].copy()
    print (ran_seqs['TAG'].unique(), ran_seqs.shape)
    subset_ran_seqs = ran_seqs[['chrom', 'start', 'end', 'ID']]
    #ran_seq_list = subset_ran_seqs.head(3).values.tolist()
    ran_seq_list = subset_ran_seqs.values.tolist()
    if DEMO:
        ran_seq_list=ran_seq_list[:3]

    captured_values = []
    for s in tqdm(ran_seq_list): 
        #print(s)
        try:
            s_in = [s[0], int(s[1]), int(s[2])]
            id_seq = s[3] 
            list_bw = eops.generate_plot_number(model, 0, interval_list=s_in, wildtype=True, 
                                            show_track=False, modify_prefix=MODIFY_PREFIX) 
        except RuntimeError:
        # Infrequent "the entries are out of order" error for some random seqs 
            continue
        try:
            out_in = eops.extract_from_position(s_in, as_dataframe=True)
            #print (out_in.shape)
            out_in = out_in.mean()
            out_in['SEQ_ID'] = id_seq
            out_in['TARGET_NAME'] = 'ITSELF'
            columns_names = out_in.copy()
            captured_values.append( out_in  )
        except ValueError:
        # Infrequent "All arrays must be of the same length" error
            continue

    df_out = pd.DataFrame([x.values.tolist() for x in captured_values], columns=out_in.index)

    df_out.to_csv('TRAINING_TEST_VALIDATION_SEQS.TXT', sep='\t', index=None)
    ## Generated
    print('Remove all head 3')
    eops = EnformerOps()



    load_tracks_eops(eops) #loading the tracks

    ran_seqs = all_data[all_data['TAG'] == 'GENERATED'].copy()
    subset_ran_seqs = ran_seqs[['SEQUENCE', 'ID']]
    print ( all_data[all_data['TAG'] == 'GENERATED'].shape)
    print (ran_seqs['TAG'].unique())

    #ran_seq_list = subset_ran_seqs.head(3).values.tolist()
    ran_seq_list = subset_ran_seqs.values.tolist()
    if DEMO:
        ran_seq_list=ran_seq_list[:3]

    ENHANCER_REGION =  ['chrX', 48782929, 48783129]
    GENE_REGION = ['chrX', 48786486, 48794411]  # 48786486,48794411
    GENE_NAME = 'GATA1'

    captured_values = []
    captured_values_target = []
    for s in tqdm(ran_seq_list): 
        
        try:
            s_in = s[1]
            id_seq = s[0]   # aways inject a sequence
            #print (id_seq)
            eops.load_data([[id_seq]]) # This will be always zero (sequence passed using insert_seq_directly)
            list_bw = eops.generate_plot_number(model, -1, interval_list=ENHANCER_REGION, wildtype=False, 
                                            show_track=False, modify_prefix=MODIFY_PREFIX) 
        except RuntimeError:
        # Infrequent "the entries are out of order" error for some random seqs 
            continue
        try:
            out_in = eops.extract_from_position(ENHANCER_REGION, as_dataframe=True)
            #print (out_in.shape)
            out_in = out_in.mean()
            out_in['SEQ_ID'] = id_seq
            out_in['TARGET_NAME'] = 'ENH_GATA1'
            columns_names = out_in.copy()
            captured_values.append( out_in  )
            
            out_in = eops.extract_from_position(GENE_REGION, as_dataframe=True)
            #print (out_in.shape)
            out_in = out_in.mean()
            out_in['SEQ_ID'] = id_seq
            out_in['TARGET_NAME'] = 'GATA1'
            columns_names = out_in.copy()
            captured_values_target.append( out_in  )
            
        except ValueError:
        # Infrequent "All arrays must be of the same length" error
            continue

    df_out_ENH = pd.DataFrame([x.values.tolist() for x in captured_values], columns=['ENHANCER_' + x for x in   out_in.index])
    df_out_GENE = pd.DataFrame([x.values.tolist() for x in captured_values_target], columns=['GENE_' + x for x in   out_in.index])

    df_out = pd.concat([df_out_ENH, df_out_GENE], axis=1)

    df_out.to_csv('GENERATED_SEQS.TXT', sep='\t', index=None)
    df_out_ENH
    df_out_GENE

if __name__=="__main__":
    # import ipdb; ipdb.set_trace()
    main()
