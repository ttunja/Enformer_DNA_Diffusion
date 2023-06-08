# Setup

# from google.colab import drive
# drive.mount('/content/drive')
#!pip install papermill
# !pip install kipoiseq==0.5.2 
#!pip install igv_notebook
# !pip install pyBigWig
# !pip install joblib
# !pip install seaborn
# !pip install matplotlib
# pip install tensorflow-hub
# !pip install imagio
# !pip install pyfaidx
# !pip install tqdm
#!pip install wget
# !apt-get install bedtools
# !pip install pybedtools
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


# assert tf.config.list_physical_devices('GPU'), 'Start the colab kernel with GPU: Runtime -> Change runtime type -> GPU'

# tf.config.list_physical_devices('GPU')
#%cd /content/drive/MyDrive


# !mkdir "Enformer_Experiments"
# #%cd Enformer_Experiments/
# !mkdir data

# transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = 'data/genome.fa'
# clinvar_vcf = 'data/clinvar.vcf.gz'
targets_txt = 'https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt'
df_targets = pd.read_csv(targets_txt, sep='\t')
df_targets[df_targets.apply(lambda x : '' in x['description'] and  'CAGE' in x['description'] , 1)]
# file_url = 'https://github.com/pinellolab/DNA-Diffusion/raw/main/src/dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt'
# train_data_path = 'data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt'
# random_data_path = 'data/random_regions_train_generated_genome_10k.txt'
# Data downloads 
# (uncomment if data is not yet in your local Drive)
# 

# !wget -O - http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_file}
# !ls data
# !wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz -O data/clinvar.vcf.gz
# !wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes
# !wget https://www.dropbox.com/s/a9ggrhn3626x0di/DNA_DIFFUSION_ALL_SEQS.txt?dl=1 -O  DNA_DIFFUSION_ALL_SEQS.txt
# !wget https://www.encodeproject.org/files/ENCFF413AHU/@@download/ENCFF413AHU.bigWig
# !wget https://www.encodeproject.org/files/ENCFF093VXI/@@download/ENCFF093VXI.bigWig
# !wget https://github.com/pinellolab/DNA-Diffusion/raw/main/src/dnadiffusion/data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt -O data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt
# !wget https://www.dropbox.com/s/oqpn784x34f6pcq/random_regions_train_generated_genome_10k.txt?dl=1 -O data/random_regions_train_generated_genome_10k.txt
# !wget https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/hg38_liftover+new_CAGE_peaks_phase1and2.bed.gz
df_sizes = pd.read_table('data/hg38.chrom.sizes', header=None).head(22)

# Enformer TensorFlow base code
SEQUENCE_LENGTH = 393216

# class Enformer:

#   def __init__(self, tfhub_url):
#     self._model = hub.load(tfhub_url).model

#   def predict_on_batch(self, inputs):
#     predictions = self._model.predict_on_batch(inputs)
#     return {k: v.numpy() for k, v in predictions.items()}

#   @tf.function
#   def contribution_input_grad(self, input_sequence,
#                               target_mask, output_head='human'):
#     input_sequence = input_sequence[tf.newaxis]

#     target_mask_mass = tf.reduce_sum(target_mask)
#     with tf.GradientTape() as tape:
#       tape.watch(input_sequence)
#       prediction = tf.reduce_sum(
#           target_mask[tf.newaxis] *
#           self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

#     input_grad = tape.gradient(prediction, input_sequence) * input_sequence
#     input_grad = tf.squeeze(input_grad, axis=0)
#     return tf.reduce_sum(input_grad, axis=-1)


# class EnformerScoreVariantsRaw:

#   def __init__(self, tfhub_url, organism='human'):
#     self._model = Enformer(tfhub_url)
#     self._organism = organism
  
#   def predict_on_batch(self, inputs):
#     ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
#     alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

#     return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


# class EnformerScoreVariantsNormalized:

#   def __init__(self, tfhub_url, transform_pkl_path,
#                organism='human'):
#     assert organism == 'human', 'Transforms only compatible with organism=human'
#     self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
#     with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
#       transform_pipeline = joblib.load(f)
#     self._transform = transform_pipeline.steps[0][1]  # StandardScaler.
    
#   def predict_on_batch(self, inputs):
#     scores = self._model.predict_on_batch(inputs)
#     return self._transform.transform(scores)


# class EnformerScoreVariantsPCANormalized:

#   def __init__(self, tfhub_url, transform_pkl_path,
#                organism='human', num_top_features=500):
#     self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
#     with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
#       self._transform = joblib.load(f)
#     self._num_top_features = num_top_features
    
#   def predict_on_batch(self, inputs):
#     scores = self._model.predict_on_batch(inputs)
#     return self._transform.transform(scores)[:, :self._num_top_features]
# class FastaStringExtractor:
    
#     def __init__(self, fasta_file):
#         self.fasta = pyfaidx.Fasta(fasta_file)
#         self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

#     def extract(self, interval: Interval, **kwargs) -> str:
#         # Truncate interval if it extends beyond the chromosome lengths.
#         chromosome_length = self._chromosome_sizes[interval.chrom]
#         trimmed_interval = Interval(interval.chrom,
#                                     max(interval.start, 0),
#                                     min(interval.end, chromosome_length),
#                                     )
#         # pyfaidx wants a 1-based interval
#         sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
#                                           trimmed_interval.start + 1,
#                                           trimmed_interval.stop).seq).upper()
#         # Fill truncated values with N's.
#         pad_upstream = 'N' * max(-interval.start, 0)
#         pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
#         return pad_upstream + sequence + pad_downstream

#     def close(self):
#         return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}
def plot_tracks(tracks, interval, height=1.5, color='blue',set_y=False):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y, color=color)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  #plt.tight_layout()
  if set_y:
    plt.ylim(set_y[0],set_y[1])
model = Enformer(model_path)

fasta_extractor = FastaStringExtractor(fasta_file)
# EnformerOps
class EnformerOps:
    def __init__(self):
        self.tracks = []
        self.input_sequences_file_path = None
        self.interval_list = None
        self.capture_bigwig_names = None
        self.full_generated_range_start = None
        self.full_generated_range_end = None
        self.loaded_seqs = None
    

    def add_track(self, track):
        """
        Adds a track to the list of tracks to be visualized.

        Args:
            track (dict): A dictionary specifying the track to be added. 
            Should have the keys "name", "file", "color", "type", and "id" 
            (if type is "enformer").
        """
        self.tracks.append(track)

    def remove_track(self, track_key):
      """
      Removes track from track list.

      Args:
          track_key (str or list): Name of the track(s) to be deleted
      """

      if type(track_key) == str:
        self.tracks.remove(track_key)
      elif type(track_key) == list:
        for key in track_key:
          self.tracks.remove(track_key)
      else:
        raise TypeError("track_key must be of type str or list")

    def load_data(self, input_sequences_file_path):
      
        if type(input_sequences_file_path) == list:
          self.loaded_seqs =  [ [x] for x in input_sequences_file_path]
          self.input_sequences_file_path = input_sequences_file_path
        

    def generate_plot_number(self, 
                             sequence_number_thousand, 
                             step=-1, 
                             interval_list=None,
                             show_track=True, 
                             capture_bigwig_names=True,
                             wildtype=False,
                            insert_seq_directly=False,
                            modify_prefix=''):
        """
        Generates IGV tracks for a given sequence in a diffusion dataset.

        Args:
            sequence_number_thousand (int): The number of the sequence ID in 
            the diffusion sequences FASTA dataset.

            step (int, optional): Which diffusion step to use. Default is -1, 
            which means the last diffusion step (i.e., the final diffused sequence).
            
            interval_list (list, optional): Coordinate to insert the 200 bp 
            sequence. Should be in BED format (chr, start, end). Default is None.
            
            show_track (bool, optional): Whether to generate IGV tracks as a result. 
            Default is True.
            
            capture_bigwig_names (bool, optional): Whether to output a list with
            all IGV tracks generated and used (in case real bigwig files were used)
            for the final visualization. Default is True.

            wildtype (bool, False)
            Dont insert and capture the wildtype sequence
        Returns:
            list: A list with the name of all bigwig files generated.
        """
        capture_bigwig_names = [] # return the name of all bigwig 
        USE_INTERVAL = interval_list
        if not interval_list:
            USE_INTERVAL = self.interval_list # this should be your 200 bp region

        if USE_INTERVAL is None:
            raise ValueError("Interval list must be specified.")

        target_interval = kipoiseq.Interval(USE_INTERVAL[0], USE_INTERVAL[1], USE_INTERVAL[2])

        chr_test = target_interval.resize(SEQUENCE_LENGTH).chr
        start_test = target_interval.resize(SEQUENCE_LENGTH).start
        end_test = target_interval.resize(SEQUENCE_LENGTH).end

        seq_to_mod = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))

        all_seqs_test = self.loaded_seqs[sequence_number_thousand]


        SEQ_IN = self.insert_seq(all_seqs_test[step], seq_to_mod, dont_insert=wildtype) # JUST THE LAST
        predictions = self.predict_from_sequence(SEQ_IN)

        mod_start = int(start_test + ((end_test - start_test)/2)) - int(114688/2)
        mod_end = int(start_test + ((end_test - start_test)/2)) + int(114688/2)


        self.full_generated_range_start = mod_start
        self.full_generated_range_end = mod_end 
        self.full_generated_chr = chr_test


        if show_track:
            igv_notebook.init()
            b = igv_notebook.Browser({
                "genome": "hg38",
                "locus": f"{chr_test}:{mod_start}-{mod_end}"
            })
        
        for track in self.tracks:
            #print (track)
            if track['type'] == 'enformer':
                id = track['id']
                n = modify_prefix + track['name']
                lg = track['log']

                p_values = predictions[:, id]
                if lg == True:
                    p_values =np.log10(1 + predictions[:, id])
                capture_bigwig_names.append(n+'.bw')
                out_track = self._enformer_bigwig_creation(chr_test, mod_start, p_values, n) # change this pretiction t/name for a real thing
                if show_track:
                    b.load_track(out_track)

            elif track['type'] == 'real':
                n = track['name']
                f = modify_prefix + track['file']
                c = track['color']
                capture_bigwig_names.append(f)
                if show_track:
                    b.load_track(self._generate_real_tracks(n, f, c))

        self.capture_bigwig_names = capture_bigwig_names

        return capture_bigwig_names


    def capture_full_cords(self):
      if self.full_generated_range_start:
        return  self.full_generated_chr , self.full_generated_range_start, self.full_generated_range_end 
      else:
        print ('Run generate_plot_number before it')


    def extract_from_position(self, position, as_dataframe=False):
        """
        Extracts data from the bigwig files generated by generate_plot_number for a given genomic region.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the region.
            end (int): The end position of the region.

        Returns:
            list: A list of dictionaries containing the name of each bigwig file 
            and the values for the given region.
        """
        if self.capture_bigwig_names is None:
            raise ValueError("Must call generate_plot_number first to generate the bigwig files.")

        results = []

        for name in self.capture_bigwig_names:
            bw = pyBigWig.open(name)
            values = bw.values(position[0], position[1], position[2])
            results.append({
                'name': name,
                'values': values
            })
        if as_dataframe:
          results = pd.DataFrame({ k['name']: k['values'] for k in results })

        return results

    @staticmethod
    def predict_from_sequence(input_sequence):
      sequence_one_hot = one_hot_encode(input_sequence)
      return model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]

    @staticmethod
    def insert_seq(seq_x, seq_mod_in, dont_insert=False):
        '''
        This function inserts a sequence `seq_x` into a larger sequence `seq_mod_in`.
        
        Args:
            seq_x (str): The sequence to be inserted into `seq_mod_in`.
            seq_mod_in (str): The larger sequence that `seq_x` will be inserted into.
            dont_insert (bool, optional): Whether or not to skip inserting `seq_x`. 
            If `True`, `seq_mod_in` will be returned unchanged. Default is `False`.
        
        Returns:
            str: The modified sequence with `seq_x` inserted into `seq_mod_in`.
        '''
        seq_to_mod_array = np.array(list(seq_mod_in))
        seq_mod_center = seq_to_mod_array.shape[0] // 2
        if not dont_insert:
            seq_to_mod_array[seq_mod_center - 100:seq_mod_center + 100] = np.array(list(seq_x))
        # else:
            # print('Keeping endogenous sequence...')
        return ''.join(seq_to_mod_array)

    @staticmethod
    def _enformer_bigwig_creation(chr_name, start, values, track_name, color='BLUE'):
        """
        Creates a bigwig file for an Enformer track.

        Args:
            chr_name (str): The name of the chromosome.
            start (int): The start position of the track.
            values (np.array): The values to be used in the track.
            track_name (str): The name of the track.
            color (str, optional): The color to use for the track. Default is 'BLUE'.

        Returns:
            dict: A dictionary containing the name and path of the bigwig file,
            as well as its format, display mode, and color.
        """
    
        t_name = f"{track_name}.bw"
        with open(t_name, 'w') as f:
            pass
        bw = pyBigWig.open(t_name, "w")
        bw.addHeader([(chr_name, coord) for chr_name, coord in df_sizes.values])
        values_conversion = (values * 1000 ).astype(np.int64) + 0.0
        bw.addEntries(chr_name, [start + (128 * x) for x in range(values_conversion.shape[0])], values=values_conversion, span=128)

        return {
            "name": f"{track_name}",
            "path": f"{track_name}.bw",
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": f"{color}",
            "height": 100,
        }

    
    def _generate_real_tracks(self, name, filename, color):
        """
        Generates a real track for a given bigwig file.

        Args:
            name (str): The name of the track.
            file (str): The name of the bigwig file to use for the track.
            color (str): The color to use for the track.


        Returns:
            dict: A dictionary containing the name, path, format, display mode, and color of the track.
        """


        chr_name, start, end = self.capture_full_cords()
        t_name = f"{name}_minimal.bw"
        with open(t_name, 'w') as f:
            pass
        bw = pyBigWig.open(t_name, "w")
        bw.addHeader([(chr_name, coord) for chr_name, coord in df_sizes.values])
        bw_cut = pyBigWig.open(filename, "r")
        values = np.array(bw_cut.values(chr_name, start, end))
        
        values_conversion = (values * 1000 ).astype(np.int64) + 0.0
        print (values_conversion)
        print (chr_name, start)
        bw.addEntries(chr_name, [r for r in  range(start, start+len(values_conversion)) ] , values=list(values_conversion), span=1, step =1)
        bw.close()
        bw_cut.close()


        return {
            "name": name,
            "path": t_name,
            "format": "bigwig",
            "displayMode": "EXPANDED",
            "color": color,
            "height": 100,
            }

    def tiling(self, interval_to_window, window=2000, slice=200):
            slice_len = int(((interval_to_window[2] + window) - (interval_to_window[1] - window)) / slice)
            start_slice = (interval_to_window[1] - window)
            slices_position = [[interval_to_window[0], start_slice + (slice * n), start_slice + ((slice * n) + slice)] for n in range(slice_len)]
            return slices_position

    def generate_tiling(self, coord_to_tile, gata_gene_region):
        #TODO df_sizes currently hardcoded since I am not sure what this is
        df_sizes = pd.read_table('hg38.chrom.sizes', header=None).head(22)
        tiling_coords = self.tiling(coord_to_tile, window=2000)
        regions_capture = []

        t_name = "tiling_vis_"+str(coord_to_tile[1])+"_"+str(coord_to_tile[2])+".bw"
        if os.path.exists(t_name):
            os.remove(t_name)
        with open(t_name, 'w') as f:
            pass
        bw_insert = pyBigWig.open(t_name, "w")
        bw_insert.addHeader([(chr, coord) for chr, coord in df_sizes.values])

        for t in tqdm(tiling_coords):
            bw_list = self.generate_plot_number(120, 1, interval_list=t, show_track=False)
            return_bw_by_tile = self.extract_from_position(gata_gene_region)
            mean_values_region_cage = np.mean(return_bw_by_tile[1]['values']).astype(np.int64) + 0.0
            bw_insert.addEntries(t[0], [t[1]], values=[mean_values_region_cage], span=200)
            regions_capture.append(mean_values_region_cage)
        
        bw_insert.close()
        return regions_capture
import pandas as pd
from IPython.display import HTML, display

class SEQ_EXTRACT:
  def __init__(self, data):
    self.data= pd.read_csv(data, sep='\t')

  def extract_seq (self, tag, cell_type):
    return self.data.query(f'TAG == "{tag}" and CELL_TYPE	 == "{cell_type}" ').copy()
  def __repr__(self):
    display(self.data.groupby(['TAG', 'CELL_TYPE']).count())
    return  'Data structure'

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
        
        # eops.add_track({
        #     'name': 'CAGE:hESC_enformer',
        #     'file': None,
        #     'color': 'RED',
        #     'type': 'enformer',
        #     'id': 500,
        #     'log': True
        # })

        # eops.add_track({
        #     'name': 'K562 DNASE REAL',
        #     'file': 'ENCFF413AHU.bigWig',
        #     'color': 'skyblue',
        #     'type': 'real'
        # })
        # eops.add_track({
        #     'name': 'GM12878 DNASE REAL',
        #     'file': 'ENCFF093VXI.bigWig',
        #     'color': 'skyblue',
        #     'type': 'real'
        # })

        # eops.remove_track(['CAGE:hESC_enformer', 'DNAse:hESC_enformer'])
    # !pwd
    # Enformer output analysis of training; random; and generated sequences
    # all_data
    # 6 hours  random
    # 20 hours promoters 
    # 50 hours train/test/validation
    # 350 hours  GENERATED
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
    #ran_seq_list = subset_ran_seqs.head(3).values.tolist() # removing demo

    ran_seq_list = subset_ran_seqs.values.tolist() # removing demo
    if DEMO:
        ran_seq_list=ran_seq_list[:3]

    captured_values = []
    for s in ran_seq_list: 
        #print(s)
        try:
            s_in = [s[0], int(s[1]), int(s[2])]
            id_seq = s[3] 
            list_bw = eops.generate_plot_number(0, interval_list=s_in, wildtype=True, 
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
            list_bw = eops.generate_plot_number(0, interval_list=s_in, wildtype=True, 
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
            list_bw = eops.generate_plot_number(0, interval_list=s_in, wildtype=True, 
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
            list_bw = eops.generate_plot_number(-1, interval_list=ENHANCER_REGION, wildtype=False, 
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
    import ipdb; ipdb.set_trace()
    main()
