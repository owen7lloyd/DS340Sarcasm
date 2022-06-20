import json
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, MutableSequence, Optional, Sequence, Tuple

import h5py
import jsonlines
import numpy as np
from sklearn.model_selection import StratifiedKFold
from settings import Settings

CLS_TOKEN_INDEX = 0

def load_pickle(filename):
    with open(filename, "rb") as file:
        return pickle.load(file, encoding="latin1")

settings = Settings()

class DataLoad:
    DATA_PATH = "data/sarcasm_data.json"
    AUDIO_PICKLE = "data/audio_new.p"
    INDICES_FILE = "data/split_indices.p"
    BERT_TARGET_EMBEDDINGS = "data/bert-output.jsonl"
    BERT_CONTEXT_EMBEDDINGS = "data/bert-output-context.jsonl"
    SHOW_ID = 9

    def __init__(self):
        self.settings = settings

        with open(self.DATA_PATH) as file:
            dataset_dict = json.load(file)

        if settings.use_target_text:
            text_bert_embeddings = []
            with jsonlines.open(self.BERT_TARGET_EMBEDDINGS) as utterances:
                for utterance in utterances:
                    features = utterance["features"][CLS_TOKEN_INDEX]
                    bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                                     for layer in range(4)], axis=0)
                    text_bert_embeddings.append(np.copy(bert_embedding_target))
        else:
            text_bert_embeddings = None

        if settings.use_context:
            context_bert_embeddings = self.load_context_bert(dataset_dict)
        else:
            context_bert_embeddings = None

        if settings.use_target_audio:
            audio_features = load_pickle(self.AUDIO_PICKLE)
        else:
            audio_features = None

        if settings.use_target_video:
            video_features_file = h5py.File("data/features/utterances_final/normal/resnet_pool5.hdf5")
            context_video_features_file = h5py.File("data/features/context_final/resnet_pool5.hdf5")
            # context_video_features_file = None
        else:
            video_features_file = None
            context_video_features_file = None

        self.data_input = []
        self.data_output = []

        self.train_ind_si = []
        self.test_ind_si = []

        self.parse_data(dataset_dict, audio_features, video_features_file, context_video_features_file,
                        text_bert_embeddings, context_bert_embeddings)

        if settings.use_target_video:
            video_features_file.close()
            context_video_features_file.close()

        self.split_indices = None
        self.stratified_k_fold()

        self.speaker_independent_split()

    def parse_data(self, dataset_dict, audio_features, video_features_file, context_video_features_file, text_embeddings, context_embeddings):
        for idx, id_ in enumerate(dataset_dict):
            self.data_input.append((dataset_dict[id_]["utterance"], dataset_dict[id_]["speaker"],
                                    dataset_dict[id_]["context"], dataset_dict[id_]["context_speakers"],
                                    audio_features[id_] if audio_features else None,
                                    video_features_file[id_][()] if video_features_file else None,
                                    context_video_features_file[id_][()] if context_video_features_file else None,
                                    text_embeddings[idx] if text_embeddings else None,
                                    context_embeddings[idx] if context_embeddings else None,
                                    dataset_dict[id_]["show"]))
            self.data_output.append(int(dataset_dict[id_]["sarcasm"]))

    def load_context_bert(self, dataset):
        length = [len(dataset[id_]["context"]) for id_ in dataset]

        with jsonlines.open(self.BERT_CONTEXT_EMBEDDINGS) as utterances:
            context_utterance_embeddings = []
            for utterance in utterances:
                features = utterance["features"][CLS_TOKEN_INDEX]
                bert_embedding_target = np.mean([np.array(features["layers"][layer]["values"])
                                                 for layer in [0, 1, 2, 3]], axis=0)
                context_utterance_embeddings.append(np.copy(bert_embedding_target))
        cumulative_length = [length[0]]
        cumulative_value = length[0]
        for val in length[1:]:
            cumulative_value += val
            cumulative_length.append(cumulative_value)
        end_index = cumulative_length
        start_index = [0] + cumulative_length[:-1]

        return [[context_utterance_embeddings[idx] for idx in range(start, end)]
                for start, end in zip(start_index, end_index)]

    def stratified_k_fold(self, splits = 10):
        cross_validator = StratifiedKFold(n_splits=splits, shuffle=True)
        split_indices = [(train_index, test_index)
                         for train_index, test_index in cross_validator.split(self.data_input, self.data_output)]

        if not os.path.exists(self.INDICES_FILE):
            with open(self.INDICES_FILE, "wb") as file:
                pickle.dump(split_indices, file, protocol=2)

    def get_stratified_k_fold(self):
        self.split_indices = load_pickle(self.INDICES_FILE)
        return self.split_indices

    def speaker_independent_split(self):
        for i, data in enumerate(self.data_input):
            if data[1] == "CHANDLER":
                self.test_ind_si.append(i)
            else:
                self.train_ind_si.append(i)

    def get_speaker_independent(self):
        return self.train_ind_si, self.test_ind_si

    def get_split(self, indices):
        data_input = [self.data_input[i] for i in indices]
        data_output = [self.data_output[i] for i in indices]
        return data_input, data_output

class Helper:
    UTT_ID = 0
    SPEAKER_ID = 1
    CONTEXT_ID = 2
    CONTEXT_SPEAKERS_ID = 3
    TARGET_AUDIO_ID = 4
    TARGET_VIDEO_ID = 5
    CONTEXT_VIDEO_ID = 6
    TEXT_BERT_ID = 7
    CONTEXT_BERT_ID = 8

    def __init__(self, train_input, train_output,
                 test_input, test_output, data):
        self.data = data
        self.config = settings
        self.train_input = train_input
        self.train_output = train_output
        self.test_input = test_input
        self.test_output = test_output

        self.author_to_index = None
        self.UNK_AUTHOR_ID = None
        self.audio_max_length = None

    @staticmethod
    def clean_str(s):
        s = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\"", " \" ", s)
        s = re.sub(r"\(", " ( ", s)
        s = re.sub(r"\)", " ) ", s)
        s = re.sub(r"\?", " ? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r"\.", " . ", s)
        s = re.sub(r"., ", " , ", s)
        s = re.sub(r"\\n", " ", s)
        return s.strip().lower()

    def get_data(self, id_, mode):
        if mode == "train":
            return [instance[id_] for instance in self.train_input]
        elif mode == "test":
            return [instance[id_] for instance in self.test_input]
        else:
            raise ValueError(f"Unrecognized mode: {mode}")

    def get_target_bert_feature(self, mode):
        return self.get_data(self.TEXT_BERT_ID, mode)

    def get_context_bert_features(self, mode):
        utterances = self.get_data(self.CONTEXT_BERT_ID, mode)
        return np.array([np.mean(utterance, axis=0) for utterance in utterances])

    def get_author(self, mode):
        authors = self.get_data(self.SPEAKER_ID, mode)

        if mode == "train":
            author_set = {"PERSON"}

            for author in authors:
                author = author.strip()
                if "PERSON" not in author:
                    author_set.add(author)

            self.author_to_index = {author: i for i, author in enumerate(author_set)}
            self.UNK_AUTHOR_ID = self.author_to_index["PERSON"]
            self.config.num_authors = len(self.author_to_index)

        authors = [self.author_to_index.get(author.strip(), self.UNK_AUTHOR_ID) for author in authors]
        return self.to_one_hot(authors, len(self.author_to_index))

    def one_hot_output(self, mode, size):
        if mode == "train":
            return self.to_one_hot(self.train_output, size)
        elif mode == "test":
            return self.to_one_hot(self.test_output, size)
        else:
            raise ValueError("Set mode properly for toOneHot method() : mode = train/test")

    @staticmethod
    def to_one_hot(data, size):
        one_hot_data = np.zeros((len(data), size))
        one_hot_data[range(len(data)), data] = 1

        assert np.array_equal(data, np.argmax(one_hot_data, axis=1))
        return one_hot_data

    @staticmethod
    def get_audio_max_length(data):
        return max(feature.shape[1] for feature in data)

    @staticmethod
    def pad_audio(data, max_length):
        for i, instance in enumerate(data):
            if instance.shape[1] < max_length:
                instance = np.concatenate([instance, np.zeros((instance.shape[0], (max_length - instance.shape[1])))],
                                          axis=1)
                data[i] = instance
            data[i] = data[i][:, :max_length]
            data[i] = data[i].transpose()
        return np.array(data)

    def get_target_audio(self, mode):
        audio = self.get_data(self.TARGET_AUDIO_ID, mode)

        if mode == "train":
            self.audio_max_length = self.get_audio_max_length(audio)

        audio = self.pad_audio(audio, self.audio_max_length)

        if mode == "train":
            self.config.audio_length = audio.shape[1]
            self.config.audio_embedding = audio.shape[2]

        return audio

    def get_target_audio_pool(self, mode):
        audio = self.get_data(self.TARGET_AUDIO_ID, mode)
        return np.array([np.mean(feature_vector, axis=1) for feature_vector in audio])

    def get_target_video_pool(self, mode):
        video = self.get_data(self.TARGET_VIDEO_ID, mode)
        return np.array([np.mean(feature_vector, axis=0) for feature_vector in video])