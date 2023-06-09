import tensorflow_hub as hub
import joblib
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import tensorflow as tf

class Enformer:

    def __init__(self, tfhub_url):
        self._model = hub.load(tfhub_url).model

    def predict_on_batch(self, inputs):
        predictions = self._model.predict_on_batch(inputs)
        return {k: v.numpy() for k, v in predictions.items()}

    @tf.function
    def contribution_input_grad(self, input_sequence,
                                target_mask, output_head='human'):
        input_sequence = input_sequence[tf.newaxis]

        target_mask_mass = tf.reduce_sum(target_mask)
        with tf.GradientTape() as tape:
            tape.watch(input_sequence)
        prediction = tf.reduce_sum(
            target_mask[tf.newaxis] *
            self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

        input_grad = tape.gradient(prediction, input_sequence) * input_sequence
        input_grad = tf.squeeze(input_grad, axis=0)
        return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

    def __init__(self, tfhub_url, organism='human'):
        self._model = Enformer(tfhub_url)
        self._organism = organism
    
    def predict_on_batch(self, inputs):
        ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
        alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

        return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                organism='human'):
        assert organism == 'human', 'Transforms only compatible with organism=human'
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            transform_pipeline = joblib.load(f)
        self._transform = transform_pipeline.steps[0][1]  # StandardScaler.
        
    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

    def __init__(self, tfhub_url, transform_pkl_path,
                organism='human', num_top_features=500):
        self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
        with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
            self._transform = joblib.load(f)
        self._num_top_features = num_top_features
        
    def predict_on_batch(self, inputs):
        scores = self._model.predict_on_batch(inputs)
        return self._transform.transform(scores)[:, :self._num_top_features]

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                        trimmed_interval.start + 1,
                                        trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
