import numpy as np
import torch
from data_prep import CustomDataset, VOCAB_SIZE
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
torch.manual_seed(3407)

word_vector_size = 50
learning_rate = 0.0005
number_of_epochs = 1  # dummy
batch_size = 1


class Skipgram(object):

    def __init__(self, vocabulary_size):

        self.input_layer_size = self.output_layer_size = vocabulary_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        if torch.cuda.is_available():
            print(f'Device name {torch.cuda.get_device_name()}')
        self.clip = 0.8
        self.weights = []
        self.hidden = torch.randn(size=(word_vector_size,
                        batch_size)).to(self.device)
        self.weights.append(
            torch.randn(size=(self.input_layer_size,
                        word_vector_size)).to(self.device)
        )
        self.weights.append(
            torch.randn(size=(word_vector_size, self.output_layer_size)).to(
                self.device)
        )

    def __call__(self, X):

        # input layer -> hidden layer
        self.hidden = torch.matmul(self.weights[0].T, X)
        # hidden layer -> output layer
        u = (torch.matmul(self.weights[1].T, self.hidden))
        y = torch.nn.Softmax(u).dim
        y.reshape(X.shape[0], 1)
        return y

    def backward(self, X, t):
        # hidden layer -> output layer
        X = X.to(self.device)
        t = t.to(self.device)
        y = self(X)
        y = y.expand(y.shape[0], t.shape[1])
        e = torch.sub(y, t)
        ei = torch.sum(e, axis=1)
        ei = ei.reshape((y.shape[0], 1))
        dw = torch.clip(torch.matmul(
            self.hidden, ei.T), min=-self.clip, max=self.clip)
        self.weights[1] = self.weights[1] - \
            learning_rate * dw  # weight update rule
        # input layer -> hidden layer
        eh = torch.matmul(self.weights[1], ei).T
        dw = torch.clip(eh.expand(VOCAB_SIZE,eh.shape[1]), min=-self.clip, max=self.clip)
        self.weights[0] = self.weights[0] - \
            learning_rate * dw  # weight update rule


def loss_function(weights, h, t, C):
    sum = 0
    for i, col_t in enumerate(t.T):
        col_t = col_t.reshape((t.shape[0], 1))
        true_output_index = (col_t == 1).nonzero(as_tuple=True)[0]
        true_output_vector = weights[1].T[true_output_index].reshape(
            (h.shape[0], 1))
        sum += torch.matmul(true_output_vector.T, h).item()
    return -sum + C * torch.log(torch.sum(torch.exp(torch.matmul(weights[1].T, h)))).item()


def read_dataset():
    return


def train():
    return


def test_data_predictions(network, test_input):
    return


def main():
    vocab_size = VOCAB_SIZE
    print(f'Vocabulary size is {VOCAB_SIZE}')
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, num_workers=6)
    skipgram_model = Skipgram(vocab_size)
    i = 0
    for batch in tqdm(data_loader):
        t, x = batch
        t = t.squeeze().T.type(torch.FloatTensor)
        x = x.T.type(torch.FloatTensor)
        skipgram_model.backward(x, t)
        if i%10000 == 0:
          print(loss_function(skipgram_model.weights,skipgram_model.hidden,t,4))
        del x, t
        i += 1
    with open('../word_vectors/skipgram_wv.pickle', 'wb') as f:
        pickle.dump(skipgram_model.weights[0], f)


if __name__ == '__main__':
    main()
