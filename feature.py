import dni
import importlib
importlib.reload(dni)
import ray
# register functions instead of passing a subclass
class AbsFeatureExtractor(dni.baseops.UnaryOp):
    def get_next(self):
        raise Exception("Not implemented")
    def has_next(self):
        raise Exception("Not implemented")
    def extract(self):
        raise Exception("Not implemented")
    def intermediate(self):
        raise Exception("Not implemented")
        
class FeatureExtractor(AbsFeatureExtractor):
    def __init__(self, accessmethod, intermediate_function, list_of_featurefunctions):
        super().__init__(accessmethod)
        self.intermediate = intermediate_function
        self.features = list_of_featurefunctions
    # contained in abstract
    def get_next(self):
        if not self.has_next():
            return None    
        input = ray.get(self.c.get_next.remote())    
        return self.extract(input)
    def has_next(self):
        return ray.get(self.c.has_next.remote())
    def extract(self, input):
        intermediate = self.intermediate(input)
        features_task = []
        # apply each feature function
        for featurefunction in self.features:
                features_task.append(ray.remote(featurefunction).remote(intermediate,input))
        features = []
        for i in range(len(features_task)):
                features.append(ray.get(features_task[i]))
        return features

class OneTimeFeatureExtractor(FeatureExtractor):
    def __init__(self, accessmethod, featurefunction):
        # pass identity function for one time feature extractor
        super().__init__(self, accessmethod, featurefunction, [lambda a : a])
        
class FeatureFactoryExtractor(FeatureExtractor):
    # featurefactory receives the intermediate data, record and features name, outputs corresponding feature
    def __init__(self, accessmethod, intermediatefunction, featurefactory, names):
        # here we use partial function trick https://www.geeksforgeeks.org/partial-functions-python/
        super().__init__(self, accessmethod, intermediatefunction,[featurefactory(intermediate,record,name) for name in names])
    
#dynamicaaly build physical activation extractor based on logical activation extractor
def build_physical_feature_ext(LogicalFeatureExt):
    @ray.remote
    class PhysicalFeatureExt(LogicalFeatureExt):
        def get_next(self):
            if not self.has_next():
                return None
            input_table = ray.get(self.c.get_next.remote())
            table = dni.tool.HighDimensionPartitionableTable()
            # read the whole batch of inputs
            for num, input in input_table.itr():
                table.merge(dni.tool.features_list_to_array(self.extract(input)))    
            return table
        def has_next(self):
            return super().has_next()
    return PhysicalFeatureExt
        