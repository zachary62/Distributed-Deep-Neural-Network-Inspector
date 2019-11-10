import numpy as np
import collections

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class TwoDimEncoders:

    @staticmethod
    def raw_to_encoded(seqs, seqs2=None, cust_char2int=None):

        if cust_char2int is None:
            all_chars = reduce(lambda chars,seq : set(chars) | set(seq), seqs)
            if seqs2 is not None:
                all_chars |= reduce(lambda chars,seq : set(chars) | set(seq),
                                    seqs2)
            all_chars = sorted(all_chars)

            char2int =  {c:i for i, c in enumerate(all_chars)}
        else:
            char2int = cust_char2int

        encoded_seqs = [[char2int[x] for x in seq] for seq in seqs]
        if seqs2 is None:
            return encoded_seqs, char2int

        encoded_seqs2 = [[char2int[x] for x in seq] for seq in seqs2]
        return encoded_seqs, char2int, encoded_seqs2

    @staticmethod
    def encoded_to_bin_tensor(enc_seqs, char2int, enc_seqs2=None,
                                start_at_min=False):

        min_x = min(char2int.values()) if start_at_min else 0
        n_chars = max(char2int.values()) - min_x + 1

        seq_len = len(enc_seqs[0])

        X = np.zeros((len(enc_seqs), seq_len, n_chars), dtype=np.int)
        for i, enc_seq in enumerate(enc_seqs):
            for j, x in enumerate(enc_seq):
                k = x - min_x
                X[i, j, k] = 1

        if enc_seqs2 is None:
            return X

        X2 = np.zeros((len(enc_seqs2), seq_len, n_chars), dtype=np.int)
        for i, enc_seq in enumerate(enc_seqs2):
            for j, x in enumerate(enc_seq):
                k = x - min_x
                X2[i, j, k] = 1

        return X, X2

    @staticmethod
    def raw_to_bin_tensor(seqs, seqs2=None, cust_char2int=None):
        encoding_out = TwoDimEncoders.raw_to_encoded(seqs, seqs2, cust_char2int)
        if len(encoding_out) == 2:
            encoded_seqs, char2int = encoding_out
            encoded_seqs2 = None
        else:
            encoded_seqs, char2int, encoded_seqs2 = encoding_out

        bin_tensors = TwoDimEncoders.encoded_to_bin_tensor(
            encoded_seqs, char2int, encoded_seqs2)

        if type(bin_tensors) is not tuple:
            return bin_tensors, char2int

        out = bin_tensors + (char2int,)
        return out


class OneDimEncoders:

    @staticmethod
    def raw_to_encoded(seq, seq2=None, cust_char2int=None):

        if cust_char2int is None:
            all_chars = set(seq)
            if seq2 is not None:
                all_chars |= set(seq2)
            all_chars = sorted(all_chars)

            char2int =  {c:i for i, c in enumerate(all_chars)}
        else:
            char2int = cust_char2int

        encoded_seq = [char2int[x] for x in seq]
        if seq2 is None:
            return encoded_seq, char2int

        encoded_seq2 = [char2int[x] for x in seq2]
        return encoded_seq, char2int, encoded_seq2

    @staticmethod
    def encoded_to_bin_tensor(enc_seq, char2int, enc_seq2=None,
                                start_at_min=False, add_dim=False):

        min_x = min(char2int.values()) if start_at_min else 0
        n_chars = max(char2int.values()) - min_x + 1

        X = np.zeros((len(enc_seq), n_chars), dtype=np.int)
        for i, x in enumerate(enc_seq):
            j = x - min_x
            X[i, j] = 1

        if add_dim:
            X = X[:,np.newaxis,:]

        if enc_seq2 is None:
            return X

        X2 = np.zeros((len(enc_seq2), n_chars), dtype=np.int)
        for i, x in enumerate(enc_seq2):
            j = x - min_x
            X2[i, j] = 1

        if add_dim:
            X2 = X2[:,np.newaxis,:]

        return X, X2

    @staticmethod
    def raw_to_bin_tensor(seq, seq2=None, cust_char2int=None, add_dim=False):
        encoding_out = OneDimEncoders.raw_to_encoded(seq, seq2, cust_char2int)
        if len(encoding_out) == 2:
            encoded_seq, char2int = encoding_out
            encoded_seq2 = None
        else:
            encoded_seq, char2int, encoded_seq2 = encoding_out

        bin_tensors = OneDimEncoders.encoded_to_bin_tensor(
            encoded_seq, char2int, encoded_seq2, add_dim=add_dim)

        if type(bin_tensors) is not tuple:
            return bin_tensors, char2int

        out = bin_tensors + (char2int,)
        return out
