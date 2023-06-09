import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import numpy as np
import tensorflow as tf
from enformer import FastaStringExtractor

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
