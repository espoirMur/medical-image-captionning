import hashlib
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from gensim.models import FastText
import matplotlib.pyplot as plt


class Vocabulary(object):
    """
    A utility class that build the vocabulary from all the possible concepts
    """
    def __init__(self, concepts_path):
        """
        Initialize the vocabulary.
        :param caption_path: path to the caption file
        """
        self.concepts_path = concepts_path
        self.length = 0
        self.ids_to_concepts = {}
        self.concepts_to_ids = {}
        self.concepts_to_names = {}
        self.build_vocabulary()

    def build_vocabulary(self):
        """
        Build the vocabulary from the caption file.
        """
        concepts_names = pd.read_csv(self.concepts_path, index_col=0, sep="\t")
        self.length = concepts_names.shape[0]
        self.concepts_to_names = concepts_names.to_dict().get('concept_name')
        self.ids_to_concepts = dict(enumerate(self.concepts_to_names.keys()))
        self.concepts_to_ids = {concept: id for id, concept in self.ids_to_concepts.items()}
    
    def encode(self, tokens):
        """
        Encode a list of tokens into an numpy array where we put 1 at the indices of the tokens in the vocabulary.
        :param tokens: a list of tokens
        :return: a list of indices
        """

        concepts_indices = [self.concepts_to_ids[token] for token in tokens]
        concepts_vector = np.zeros(self.length)
        concepts_vector[concepts_indices] = 1
        return concepts_vector
    
    def decode(self, indices):
        """take a numpy array of indices and convert it to a string of labels

        Args:
            indices (_type_): _description_
        """
        captions_indices = np.nonzero(indices)[0]
        print(captions_indices, "***********************")
        concepts = [self.ids_to_concepts[idx] for idx in captions_indices]
        return concepts

    def get_names(self, concepts):
        """from a list of concepts, return the corresponding names

        Args:
            concepts (_type_): _description_
        """

        concepts_names = [self.concepts_to_names[concept] for concept in concepts]
        return concepts_names


class SequenceVocabulary:
    """
    the base sequence vocabulary for sequence to sequences tasks
    """
    def __init__(self, corpus,
                 pad_token="<pad>",
                 start_token="<sos>",
                 end_token="<eos>",
                 unknown_token="<unk>",
                 out_of_words=0):
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

        self.pad_id = 0
        self.start_id = None
        self.end_id = None
        self.unknown_id = None

        self.out_of_words = out_of_words
        self.vocabulary = Counter()
        self.tokens_to_id = dict()
        self.id_to_tokens = dict()
        self.corpus = corpus

    def reset(self):
        self.tokens_to_id = dict()
        self.id_to_tokens = dict()

        self._add_special_tokens()
        self._set_special_token_ids()
        self.vocabulary = Counter()

    def from_vocab_instance(self, vocab):
        for attr, value in vars(vocab).items():
            setattr(self, attr, value)
        self._set_special_token_ids()
        return self
    
    def detokenize(self, tokens):
        """
        detokenize a list of tokens
        """
        return " ".join(tokens)

    def read_embeddings(self, embeddings, words_to_index):
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)

        filtered_embeddings = np.zeros(len(self), embeddings.shape[1])
        mask = np.zeros(len(self), dtype=bool)
        missing = []
        for token_id, token in tqdm(self.id_to_tokens.items(), 
                                    desc="Reading embeddings......", 
                                    total=len(self.id_to_tokens.items())):
            if token not in words_to_index or token == self.unknown_token:
                sample = np.random.normal(mean, std/4)
                filtered_embeddings[token_id] = sample
                mask[token_id] = True
                missing.append(token_id)
            else:
                filtered_embeddings[token_id] = embeddings[words_to_index[token]]
        print("{} words are missing embeddings".format(len(missing)))
        return filtered_embeddings, mask, missing

    def read_fasttext(self, file):
        model = FastText.load(file)
        embeddings = np.zeros((len(self), model.vector_size))
        missing = []
        for token_id, token in tqdm(self.id_to_tokens.items(), 
                                    desc="Reading embeddings......", 
                                    total=len(self.id_to_tokens.items())):
            if token not in model.wv.vocab:
                missing.append(token)
            embeddings[token_id] = model[token]
        return embeddings, missing

    def add_token(self, token):
        index = len(self.tokens_to_id)

        if token not in self.tokens_to_id:
            self.tokens_to_id[token] = index
            self.id_to_tokens[index] = token

    def __hash__(self):
        return hashlib.sha256(
            json.dumps(self.tokens_to_id, sort_keys=True).encode()).hexdigest()

    def hash(self):
        return hashlib.sha256(
            json.dumps(self.tokens_to_id, sort_keys=True).encode()).hexdigest()

    def _set_special_token_ids(self):
        self.PAD_id = self.tokens_to_id.get(self.pad_id, 0)
        self.SOS_id = self.tokens_to_id[self.start_id]
        self.EOS_id = self.tokens_to_id[self.end_id]
        self.UNK_id = self.tokens_to_id[self.unknown_id]

    def _add_special_tokens(self):
        self.add_token(self.pad_id)
        self.add_token(self.start_id)
        self.add_token(self.end_id)
        self.add_token(self.unknown_id)

        for i in range(self.out_of_words):
            self.add_token(f"<oov-{i}>")

    def build(self, preprocess, suffix='.vocab'):
        """
        Build the vocab from a txt corpus.
        The function assumes that the txt file contains one sentence per line.
        Afterwards, write the vocab data to disk as {file}{suffix}.
        """
        vocab_file = f"vocabulary-caption.{suffix}"
        if os.path.exists(vocab_file):
            self.load_from_vocab_file(vocab_file)
        else:
            self._add_special_tokens()
            self._set_special_token_ids()
            for line in self.corpus:
                print(line)
                self.read_sequence(preprocess(line))
            self.save(vocab_file)

    def build_from_tokens(self, dataset):
        """
        Build the vocab from a list of wordlists.
        """
        self._add_special_tokens()
        self._set_special_token_ids()
        for tokens in dataset:
            self.read_sequence(tokens)
        self.build_lookup()

    def load_from_vocab_file(self, file):
        """
        Load vocabulary from a .vocab file
        """

        self.tokens_to_id = dict()
        self.id_to_tokens = dict()

        self._add_special_tokens()
        self._set_special_token_ids()
        self.vocabulary = Counter()

        for line in open(file, encoding="utf-8").readlines():
            token, count = line.split("\t")
            self.vocabulary[token] = float(count)
            self.add_token(token)

    def read_sequence(self, tokens):
        self.vocabulary.update(tokens)

    def add_special_tokens(self, tokens, sequence_length):
        tokens = tokens + [self.end_token]
        if self.start_token:
            tokens = [self.start_token] + tokens
        if sequence_length > 0:
            tokens = tokens[:sequence_length]
        return tokens

    def save(self, file):
        with open(file, "w", encoding="utf-8") as f:
            for w, k in self.vocabulary.most_common():
                f.write("\t".join([w, str(k)]) + "\n")

    def is_corrupt(self):
        return len([tok for tok, index in self.tokens_to_id.items()
                    if self.id_to_tokens[index] != tok]) > 0

    def get_tokens(self):
        return [self.id_to_tokens[key] for key in sorted(self.id_to_tokens.keys())]

    def build_lookup(self, size=None):
        self.tokens_to_id = dict()
        self.id_to_tokens = dict()

        self._add_special_tokens()
        self._set_special_token_ids()

        for word in self.vocabulary.most_common(size):
            self.add_token(word)

    def visualize_vocab(self):
        sns.distplot(list(self.vocabulary.values()), bins=50, kde=False)
        plt.show()

    def get_freqs(self):
        _sum = sum(self.vocabulary.values())
        freqs = dict()
        for i in range(len(self)):
            tok = self.id_to_tokens[i]
            freqs[tok] = self.vocabulary[tok] / _sum

        return freqs

    def get_freqs_list(self):
        freqs = self.get_freqs()
        return [freqs[self.id_to_tokens[i]] for i in range(len(self))]

    def start_id(self):
        return self.tokens_to_id[self.stat_id]

    def __len__(self):
        return len(self.tokens_to_id)

    def vectorize(self, tokens, out_of_vocab=0, return_map=False):
        """
        vectorize the vocabulary
        """
        ids = []
        out_of_vocab_to_token = {}
        token_to_out_of_vocab = {}
        for token in tokens:
            if token in self.tokens_to_id:
                ids.append(self.tokens_to_id[token])
            elif token in token_to_out_of_vocab:
                ids.append(self.token_to_id[token_to_out_of_vocab.get(token)])
            elif out_of_vocab > len(out_of_vocab_to_token):
                _oov = f"<oov-{len(out_of_vocab_to_token)}>"
                ids.append(self.tokens_to_id[_oov])
                out_of_vocab_to_token[_oov] = token
                token_to_out_of_vocab[token] = _oov
            else:
                ids.append(self.token_to_id[self.unknown_token])
        if out_of_vocab_to_token > 0 and return_map:
            return ids, out_of_vocab_to_token
        else:
            return ids
    
