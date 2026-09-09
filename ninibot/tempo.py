import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from voice.io.azure.blobmanager import AUDIO_LEN


class TfrecordManager:
    def __init__(self, configs):
        self.configs = configs

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def serialize_example(
        self,
        wav: list, # Expects numpy array or list of floats
        fob: float,
        hconfig: str,
        varroa: float,
        ts: float,
        operation_id: float,
        weights: float,
        buzzing: float,
        queenstate: float,
        source: str,
        long: float,
        lat: float,
    ):
        # Ensure wav is a numpy array of float32
        wav_array = np.array(wav).astype(np.float32)
        
        # CRITICAL CHANGE: Serialize raw numpy bytes, do NOT encode to WAV
        wav_bytes = wav_array.tobytes()

        feature = {
            "wav": self._bytes_feature(wav_bytes),
            "hive_configuration": self._bytes_feature(hconfig.encode("utf-8")),
            "fob": self._float_feature(fob),
            "buzzing": self._float_feature(buzzing),
            "varroa": self._float_feature(varroa),
            "ts": self._float_feature(ts),
            "queenstate": self._float_feature(queenstate),
            "operation_id": self._float_feature(operation_id),
            "weights": self._float_feature(weights),
            # Fix: 'source' is string in input but defined as float32 in your schema. 
            # Assuming it should be string/bytes based on your filter logic later.
            "source": self._bytes_feature(source.encode("utf-8")), 
            "longitude": self._float_feature(long),
            "latitude": self._float_feature(lat),
        }
        
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()   

    def build_and_save(self, data: pd.DataFrame, path: str):
        with tf.io.TFRecordWriter(path) as writer:
            for (
                wav,
                fob,
                varroa,
                ts,
                operation_id,
                weights,
                buzzing,
                queenstate,
                source,
                hconfig,
                long,
                lat,
            ) in (
                pbar := tqdm(
                    zip(
                        data["wave"],
                        data["fob"].fillna(-1.0),
                        data["varroa"].fillna(-1.0),
                        data["ts"].fillna(-1),
                        data["operation_id"],
                        data["weights"].fillna(1),
                        data["buzzing"].fillna(-1),
                        data["queenstate"].fillna(-1),
                        data["source"],
                        data["hive_configuration"].fillna("unknown"),
                        data["longitude"].fillna(-1.0),
                        data["latitude"].fillna(-1.0),
                    )
                )
            ):
                pbar.set_description(f"Processing audio sample")
                example = self.serialize_example(
                    np.array(wav).astype(np.float32),
                    fob,
                    hconfig.encode("utf-8"),
                    varroa,
                    ts,
                    operation_id,
                    weights,
                    buzzing,
                    queenstate,
                    source,
                    long,
                    lat,
                )
                pbar.set_description(f"Writing audio sample")
                writer.write(example)

    def load(self, path, keep_only_beetrack: bool = False, filter_buzzing_only: bool = False):
        feature_description = {
            "wav": tf.io.FixedLenFeature([], dtype=tf.string),
            "hive_configuration": tf.io.FixedLenFeature([], dtype=tf.string, default_value="unknown"),
            "fob": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
            "varroa": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
            "ts": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
            "buzzing": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
            "queenstate": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
            "operation_id": tf.io.FixedLenFeature([], tf.float32),
            "weights": tf.io.FixedLenFeature([], tf.float32, default_value=0.9),
            "source": tf.io.FixedLenFeature([], dtype=tf.string), # Changed from float32
            "longitude": tf.io.FixedLenFeature([], dtype=tf.float32),
            "latitude": tf.io.FixedLenFeature([], dtype=tf.float32),
        }

        def _parse_function(example_proto):
            features = tf.io.parse_single_example(example_proto, feature_description)
            audio = tf.io.decode_raw(features["wav"], out_type=tf.float32)
            audio = tf.reshape(audio, (AUDIO_LEN,))
            audio.set_shape((AUDIO_LEN,))
            
            features["wav"] = audio
            return features

        raw_dataset = tf.data.TFRecordDataset(path)
        processed_dataset = raw_dataset.map(
            _parse_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
        )

        if keep_only_beetrack:
            # Assuming source "0.0" was stored as string "0.0" or similar. Adjust logic as needed.
            processed_dataset = processed_dataset.filter(lambda x: tf.equal(x["source"], b"0.0")) 

        if filter_buzzing_only:
            processed_dataset = processed_dataset.filter(lambda x: tf.greater(x["buzzing"], 0.0))
            
        return processed_dataset   
