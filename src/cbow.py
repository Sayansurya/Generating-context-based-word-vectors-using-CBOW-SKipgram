import numpy as np
import torch
from torch.utils.data import DataLoader
from data_prep import CustomDataset, VOCAB_SIZE
from tqdm import tqdm
import pickle

torch.manual_seed(3407)

word_vector_size = 100  # N
learning_rate = 0.015
number_of_epochs = 3  # dummy
batch_size = 1  # dummy


class CBOW(object):

    def __init__(self, vocabulary_size):

        self.input_layer_size = self.output_layer_size = vocabulary_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip = 0.7
        print(self.device)
        if torch.cuda.is_available():
            print(f'Device name {torch.cuda.get_device_name()}')
        self.weights = []
        self.track_state = {}
        self.track_state['h'] = []
        self.weights.append(
            # W (weight matrix b/w input and hidden layer) has dimensions V x N
            torch.randn(size=(self.input_layer_size,
                        word_vector_size), dtype=torch.float32).to(self.device)
        )
        self.weights.append(
            # W' (weight matrix b/w hidden and output layer) has dimensions N x V
            torch.randn(
                size=(word_vector_size, self.output_layer_size), dtype=torch.float32).to(self.device)
        )

    def __call__(self, X):  # One training example will be a V x C matrix, where every column is the one hot representation of the context words
        # input layer -> hidden layer
        X = X.to(self.device)
        C = X.shape[1]
        # h = 1/C(W.T * (c1 + c2 + ... + cn)) has a dimension of N x 1
        h = torch.div(torch.matmul(
            self.weights[0].T, torch.sum(X, dim=1).unsqueeze(1)), C)
        self.track_state['h'].append(h)
        # hidden layer -> output layer
        # u = W'.T * h has a dimension of V x 1
        u = torch.matmul(self.weights[1].T, h)
        # y is the prediction of the possible target word
        y = torch.nn.Softmax(u).dim
        y = y.reshape(-1, 1)  # y has a dimension of V x 1
        return y

    def backward(self, X, t):  # t is the one hot representation of the actual target word
        X = X.to(self.device)
        t = t.to(self.device)
        y = self(X)
        C = X.shape[1]
        # hidden layer -> output layer
        dw = torch.clip(
            torch.matmul(self.track_state['h'][-1], torch.sub(y, t).T), min=-self.clip, max=self.clip
        )
        self.weights[1] = self.weights[1] - \
            learning_rate * dw  # weight update rule

        # input layer -> hidden layer
        dw = torch.clip(
            torch.matmul(self.weights[1], torch.sub(y, t)), min=-self.clip, max=self.clip
        )
        for i, x in enumerate(X.T):
            x = x.reshape((X.shape[0], 1))
            self.weights[0] = self.weights[0] - learning_rate * \
                torch.div(torch.matmul(x, dw.T), C)  # weight update rule


def loss_function(weights, h, t):
    true_output_index = (t == 1).nonzero(as_tuple=True)[0]
    true_output_vector = weights[1].T[true_output_index].reshape(
        (h.shape[0], 1))
    return -torch.matmul(true_output_vector.T, h).item() + torch.log(torch.sum(torch.exp(torch.matmul(weights[1].T, h)))).item()


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
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 8)
    cbow_model = CBOW(vocab_size)

    for batch in tqdm(data_loader):
        x, t = batch
        x = x.squeeze().T.type(torch.FloatTensor)
        t = t.T.type(torch.FloatTensor)
        cbow_model.backward(x, t)
    with open('../word_vectors/cbow_wv.pickle', 'wb') as f:
        pickle.dump(cbow_model.weights[1], f)


if __name__ == '__main__':
    main()
