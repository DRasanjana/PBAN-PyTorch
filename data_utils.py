import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import string
import re
import csv


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def parseXML1(data_path):
    tree = ET.ElementTree(file=data_path)
    objs = list()
    for sentence in tree.getroot():
        obj = dict()
        for item in sentence:
            if item.tag == 'text':
                obj['text'] = item.text
            elif item.tag == 'aspectTerms':
                obj['aspects'] = list()
                for aspectTerm in item:
                    if aspectTerm.attrib['polarity'] != 'conflict':
                        obj['aspects'].append(aspectTerm.attrib)
        if 'aspects' in obj and len(obj['aspects']):
            objs.append(obj)
    return objs


def parseXML(data_path):
    with open(data_path, 'r', encoding="utf8") as csvfile:
        aspectreader = csv.reader(csvfile, delimiter=',')
        j = 0
        count = 0
        input = []
        target = []
        lable = []
        examples = []
        ccc = []
        for row in aspectreader:
            if (j == 0):
                j = 1
            else:
                sent = row[0].lower()
                sent = remove_punct(sent)
                sent.replace('\d+', '')
                # sent.replace(r'\b\w\b', '').replace(r'\s+', ' ')
                # sent.replace('\s+', ' ', regex=True)
                # sent=re.sub(r"^\s+|\s+$", "", sent), sep='')
                sent = re.sub(r"^\s+|\s+$", "", sent)
                sent = re.sub(' +', ' ', sent)
                examples.append(sent)
                input.append(sent)
                # nb_aspects = int(row[1])
                aspect = row[1].lower()
                target.append(aspect)
                examples.append(aspect)
                polarity = int(row[2]) + 1
                examples.append(polarity)
                examples.append(row[3])
                examples.append(row[4])
                ccc.append(sent + "," + aspect + "," + str(polarity) + "," + row[3] + "," + row[4])
    x_text = input
    # find the same targets
    all_sentence = [s for s in x_text]
    targets_nums = [all_sentence.count(s) for s in all_sentence]
    # ccc_num= [ccc.count(s) for s in ccc]
    # for i in range(len(ccc_num)):
    #     if ccc_num[i]>1:
    #         print(str(i+2) +ccc[i])

    # i=0
    # while i<len(targets_nums):
    #     act=int(targets_nums[i])
    #     for j in range(act):
    #         if not (targets_nums[i+j]==act):
    #             print(x_text[i+j-1])
    #             print(targets_nums[i+j])
    #             print(i+j-1)
    #     i=i+j+1

    objs = []
    i = 0
    while i < len(all_sentence):
        obj = dict()
        num = targets_nums[i]
        aspects = []
        obj['text'] = all_sentence[i]
        for j in range(num):
            info = dict()
            info['term'] = examples[(i + j) * 5 + 1]
            info['polarity'] = examples[(i + j) * 5 + 2]
            info['from'] = examples[(i + j) * 5 + 3]
            info['to'] = examples[(i + j) * 5 + 4]
            aspects.append(info)
        obj['aspects'] = aspects
        objs.append(obj)
        i = i + num
    if 'aspects' in obj and len(obj['aspects']):
        objs.append(obj)
    return objs


def build_tokenizer(fnames, max_length, data_file):
    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''

    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad:  # pad_id should be zero (for Dynamic RNN)
            self.pad_word = '<pad>'
            self.pad_id = self._length
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():
            self._reverse_vocab_dict[i] = w

    def word_to_id(self, word):
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]

    def id_to_word(self, id_):
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]

    def has_word(self, word):
        return word in self._vocab_dict

    def __len__(self):
        return self._length


class Tokenizer(object):
    ''' transform text to indices '''

    def __init__(self, vocab, max_length, lower):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

    @classmethod
    def from_files(cls, fnames, max_length, lower=True):
        corpus = set()
        for fname in fnames:
            for obj in parseXML(fname):
                text_raw = obj['text']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower)

    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=self.max_length,
                                      padding=padding, truncating=truncating)

    def position_sequence(self, text, start, end, reverse=False, padding='post', truncating='post'):
        text_left = Tokenizer.split_text(text[:start])
        text_aspect = Tokenizer.split_text(text[start:end])
        text_right = Tokenizer.split_text(text[end:])
        tag_left = [len(text_left) - i for i in range(len(text_left))]
        tag_aspect = [0 for i in range(len(text_aspect))]
        tag_right = [i + 1 for i in range(len(text_right))]
        position_tag = tag_left + tag_aspect + tag_right
        if len(position_tag) == 0:
            position_tag = [0]
        if reverse:
            position_tag.reverse()
        return Tokenizer.pad_sequence(position_tag, pad_id=0, maxlen=self.max_length,
                                      padding=padding, truncating=truncating)

    @staticmethod
    def split_text(text):
        for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"",
                   "-", ":"]:
            text = text.replace(ch, " " + ch + " ")
        return text.strip().split()


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''

    def __init__(self, fname, tokenizer, target_dim):
        data = list()
        # polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in parseXML(fname):
            text = tokenizer.text_to_sequence(obj['text'])
            for aspect in obj['aspects']:
                if target_dim == 2 and aspect['polarity'] == 'neutral':
                    continue
                aspect_term = tokenizer.text_to_sequence(aspect['term'])
                position = tokenizer.position_sequence(obj['text'], int(aspect['from']), int(aspect['to']))
                # polarity = polarity_dict[aspect['polarity']]
                polarity = aspect['polarity']
                data.append({'text': text, 'aspect': aspect_term, 'position': position, 'polarity': polarity})
        self._data = data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def _load_wordvec(data_path, vocab=None):
    embedding_model = {}
    f = open(data_path, 'r', encoding="utf8")
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embedding_model[word] = coefs
    f.close()
    return embedding_model
    #
    # with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
    #     word_vec = dict()
    #     for line in f:
    #         tokens = line.rstrip().split()
    #         if tokens[0] == '<pad>' or tokens[0] == '<unk>': # avoid them
    #             continue
    #         if vocab is None or vocab.has_word(tokens[0]):
    #             word_vec[tokens[0]] = np.asarray(tokens[:-300], dtype='float32')
    #     return word_vec


def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = _load_wordvec(fname, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix


if __name__ == '__main__':
    tokenizer = build_tokenizer(
        fnames=['./datasets/train.csv', './datasets/test.csv'],
        max_length=80,
        data_file='{0}_tokenizer.dat'.format('law'))
    trainset = SentenceDataset('./datasets/train.csv', tokenizer, target_dim=3)
