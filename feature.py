import tool

class AbcExtractor():
    def __init__(self):
        pass
    def extract(self):
        pass
    def get_class(self):
        return "AbcExtractor"
    def get_name(self):
        return self.name
    
# input is one record, feature function takes one record, return one feature
class Feature_Extractor(AbcExtractor):
    def __init__(self,feature,name =""):
        self.feature = feature
        self.name = name 
    def extract(self,input):
        return self.feature(input)
    def get_class(self):
        return "Feature_Extractor"

    
# input is a list of records, feature function takes one record, return a list of features
class Batch_Feature_Extractor(Feature_Extractor):
    def extract(self,input):
        feature_list = []
        for i in range(len(input)):
            feature_list.append(self.feature(input[i]))
        return feature_list
    def get_class(self):
        return "Batch_Feature_Extractor"
    
class CachedFeatureExtractor(Feature_Extractor):
    # a set of feature functions that use intermediate data
    def __init__(self,features,intermediate = None,name ="", size = 30):
        self.intermediate = intermediate
        self.features = features
        self.intermediate = intermediate
        self.name = name 
        self.cache = tool.LRUCache(size)
    
    def setup(self, input):
        if self.intermediate is not None:
            value = self.intermediate(input)
            self.cache.set(input,value)
    
    # input data and which feature function to use
    def extract(self,input,featureidx):
        if self.intermediate is not None:
            inter = self.cache.get(input)
            # cached
            if inter is not None:
                return self.features[featureidx](inter, input)
            # not cached, recompute the intermediate data
            else:
                value = self.intermediate(input)
                self.cache.set(input,value)
                return self.features[featureidx](value, input)
        else:
            return self.features[featureidx](input)
    
    def get_class(self):
        return "Cached_Feature_Extractor"
    
# feature function takes batch of records
# class Batch_Feature_Extractor_2():
#     def __init__(self,feature):
#         self.feature = feature
#     def extract(self,input,unit):
#         pass
#     def get_class(self):
#         return "Feature_Extractor"