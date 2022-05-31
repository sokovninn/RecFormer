import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        input_tensor = torch.LongTensor([self.data[index]]) # the first dimension is minibatch size
        label = torch.LongTensor([self.labels[index]])
        return input_tensor, label

    def __len__(self):
        return len(self.data)

class EmbDataset(TorchDataset):

    def __init__(self, data, labels, vocab, unk_emb, ing_dict):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.unk_emb = unk_emb
        self.ing_dict = ing_dict

    def __getitem__(self, index):

        input_embeddings = []
        for id in self.data[index]:
          ing_str = self.ing_dict[id]
          ing_emb = []
          for word in ing_str.lower().split(" "):
            if word in self.vocab:
              ing_emb.append(self.vocab[word])
          if not ing_emb:
            ing_emb = [self.unk_emb]
          #print(ing_emb)
          ing_emb = np.mean(ing_emb, axis=0)
          #print(ing_emb.shape)
          input_embeddings.append(torch.Tensor(ing_emb))
        if not input_embeddings:
            input_tensor = torch.Tensor([self.unk_emb])
        else:
            input_tensor = torch.stack(input_embeddings, dim=1).T
        label = torch.LongTensor([self.labels[index]])
        return input_tensor, label

    def __len__(self):
        return len(self.data)


def pad_tensor(vec, length, dim, pad_symbol):
    # Credits to https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
    pad_size = list(vec.shape)
    #print(pad_size)
    pad_size[dim] = length - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size, dtype=torch.int64)], dim=dim)

class Padder:
    
    def __init__(self, dim=0, pad_symbol=0):
        self.dim = dim
        self.pad_symbol = pad_symbol
        
    def __call__(self, batch):
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        batch = list(map(lambda xy:
                    (pad_tensor(xy[0], max_len, self.dim, self.pad_symbol), xy[1]), batch))
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
        return xs, ys