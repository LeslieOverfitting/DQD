import torch 

class DatasetIterator(object):
    def __init__(self, data, batch_size, device):
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.batch_num = len(data) // batch_size
        self.residue = False
        if len(data) % batch_size != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, batch):
        q1 = torch.LongTensor([_[0] for _ in batch])
        q1_len = torch.LongTensor([_[1] for _ in batch])
        q2 = torch.LongTensor([_[2] for _ in batch])
        q2_len = torch.LongTensor([_[3] for _ in batch])
        label = torch.LongTensor([_[4] for _ in batch])
        return (q1, q1_len, q2, q2_len), label

    def __next__(self):
        if self.index == self.batch_num and self.residue:
           batch = self.data[self.index * self.batch_size:]
           self.index += 1
           batch = self._to_tensor(batch)
           return batch
        elif self.index >= self.batch_num:
            self.index = 0
            raise StopIteration
        else: 
            batch = self.data[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            batch = self._to_tensor(batch)
            self.index += 1
            return batch

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.batch_num + 1
        else:
            return self.batch_num