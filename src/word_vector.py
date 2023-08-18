import torch
import pickle
from torch.nn.functional import cosine_similarity


class WordVector():
    def __init__(self, i, wv_type):
        with open(f'../word_vectors/{i}_{wv_type}_vocab_file.pickle', 'rb') as f:
            self.vocab_to_idx = pickle.load(f)
        with open(f'../word_vectors/{i}_{wv_type}_wv.pickle', 'rb') as f:
            self.wv = pickle.load(f)
            if wv_type == 'skipgram':
                self.wv = self.wv.T
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.wv = self.wv.to(self.device)
        self.vocab_size = len(self.vocab_to_idx)
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}

    def __getitem__(self, word):
        return self.wv[:, self.vocab_to_idx[word]]

    def find_most_similar_from_word(self, word):
        word_wv = self[word].reshape(-1, 1)
        word_wv = word_wv.to(self.device)
        cosine_sim = cosine_similarity(word_wv, self.wv, dim=0)
        one_hot_word = torch.zeros((self.vocab_size))
        one_hot_word[self.vocab_to_idx[word]] = 1
        one_hot_word = one_hot_word.to(self.device)
        cosine_sim = cosine_sim - one_hot_word
        values, indices = torch.topk(cosine_sim, k=10, dim=0)

        values = values.to('cpu')
        indices = indices.to('cpu')
        words = {}
        for i in range(10):
            words[self.idx_to_vocab[indices[i].item()]] = round(
                values[i].item(), 4)
        return self.idx_to_vocab[indices[0].item()], words

    def find_most_similar_from_vector(self, vector):
        word_wv = vector.to(self.device).reshape(-1, 1)
        cosine_sim = cosine_similarity(word_wv, self.wv, dim=0)
        values, indices = torch.topk(cosine_sim, k=10, dim=0)

        values = values.to('cpu')
        indices = indices.to('cpu')
        words = {}
        for i in range(10):
            words[self.idx_to_vocab[indices[i].item()]] = round(
                values[i].item(), 4)
        return self.idx_to_vocab[indices[0].item()], words
    
    def find_k_similar_from_vector(self, vector, k=4):
        word_wv = vector.to(self.device).reshape(-1, 1)
        cosine_sim = cosine_similarity(word_wv, self.wv, dim=0)
        values, indices = torch.topk(cosine_sim, k=k, dim=0)

        values = values.to('cpu')
        indices = indices.to('cpu')
        words = {}
        for i in range(k):
            words[self.idx_to_vocab[indices[i].item()]] = round(
                values[i].item(), 4)
        return self.idx_to_vocab[indices[0].item()], words
