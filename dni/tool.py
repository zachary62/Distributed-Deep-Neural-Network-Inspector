import numpy as np
import collections      
import pandas as pd  
from typing import (
    IO,
    TYPE_CHECKING,
    AnyStr,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)
import itertools
Axes = Iterable
Dtype = Union[str, np.dtype, "ExtensionDtype"]


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

def features_list_to_array(features):
    feature_array = []
    for feature in np.transpose(np.array(features), (1, 0, 2)):
        feature_array.append([[[feature]]])
    return np.array(feature_array)

def activations_list_to_array(activations):
    activation_array = []
    for activation in np.transpose(np.array(activations), (1, 0, 3, 2))[..., np.newaxis,:]:
        activation_array.append([activation])
    return np.array(activation_array)

class InputTable():
    def __init__(self, table):
        # assume that table has the same format as the output from keras model prediction
        self.table = table
        self.input_num = len(self.table)
    # return an iterator of all the features of an input
    def itr(self):
        for i in range(self.input_num):
             yield i, self.table[i]
    def get_stat(self):
        return {"input_num":self.input_num}
    def add_table(self, table2):
        self.table = self.table + table2
        self.input_num = len(self.table)
        
class ActivationTable():
    def __init__(self, table=None):
        self.table = table
        # assume that table has the same format as the output from keras model prediction
        if table is not None:         
            self.layer_num = len(self.table)
            self.unit_num = self.table[0].shape[-1]
            self.input_num = self.table[0].shape[0]
            self.activation_shape = self.table[0][0,...,0].shape
    # return an iterator of all the features of an input
    def itr(self,input_num):
        for i in range(self.layer_num):
            for j in range(self.unit_num):
                yield i, j, self.table[i][input_num,...,j]
    def get_stat(self):
        if self.table is None:
            return "not initialized"
        return {"layer_num":self.layer_num,"unit_num":self.unit_num, "input_num":self.input_num, "activation_shape": self.activation_shape}
    def add_table(self,table2):
        table1 = self.table
        if table1 is None:
            new_table = table2
        else:
            new_table = []
            for i in range(self.layer_num):
                activation_per_layer = []
                for j in range(self.unit_num):
                    activation_per_layer.append(np.concatenate((table1[i][...,j], table2[i][...,j])))
                new_table.append(np.transpose(np.array(activation_per_layer), (1, 2, 0))) 
        self.__init__(new_table)
    
class FeatureTable():
    def __init__(self, table = None):
        self.table =  table
        if self.table is not None:
            self.feature_num = len(self.table)
            self.input_num = len(self.table[0])
            self.feature_shape = self.table[0][0].shape
    # return an iterator of all the features of an input
    def itr(self,input_num):
        for i in range(self.feature_num):
            yield i, self.table[i][input_num]
    def get_stat(self):
        if self.table is None:
            return "not initialized"
        return {"feature_num":self.feature_num, "input_num":self.input_num, "feature_shape": self.feature_shape}
    def add_table(self,new_table):
        if self.table is None:
            self.__init__(new_table)
        else:
            for i in range(self.feature_num):
                self.table[i] = np.concatenate((self.table[i], new_table[i]))
            self.feature_num = len(self.table)
            self.input_num = len(self.table[0])
            self.feature_shape = self.table[0][0].shape
            



class DniDataFrame():
    def __init__(self,
        data=None,
        index: Optional[Axes] = None,
        columns: Optional[Axes] = ["input_id","model_id","layer_id", "unit_id", "feature_id", "data"],
        dtype: Optional[Dtype] = None,
        copy: bool = False,
        ):
        if isinstance(data, np.ndarray):
            new_data = []
            shape = [data.shape[i] for i in range(5)]
            for idx in itertools.product(*[range(s) for s in shape]):
                data_slice = list(idx)
                data_slice.append(data[idx])
                new_data.append(data_slice)
            data = new_data       
        self._data = pd.DataFrame(data,index,columns,dtype,copy)
        self._data.set_index("input_id", inplace=True)
    # iterate through specified input_id
    def itr(self, input_id):
        i = input_id
        # given input_id, for each row
        for j in range(self._data.loc[i].shape[0]):
            row = self._data.loc[i].iloc[j]
            yield [i,row[0],row[1],row[2],row[3]],row[4]
    # partition DniDataFrame by index
    def partition(self, partition_num):
        total_num = self.index.max() - self.index.min()
        step = total_num//partition_num
        Tables = []
        for i in range(partition_num - 1):
            Tables.append(self._data.loc[self._data.index.min()+i*step:self._data.index.min()+(i+1)*step - 1])
        Tables.append(self._data.loc[self._data.index.min()+(partition_num - 1)*step:self._data.index.max()])
        return Tables
    def summary(self):
        return {"feature_num":int(self._data.describe()["feature_id"].iloc[7] + 1), "input_num":self._data.index.max() + 1, "shape": self._data.iloc[0]['data'].shape, "model_num":int(self._data.describe()["model_id"].iloc[7] + 1),"unit_num":int(self._data.describe()["unit_id"].iloc[7] + 1), "layer_num":int(self._data.describe()["layer_id"].iloc[7] + 1)}