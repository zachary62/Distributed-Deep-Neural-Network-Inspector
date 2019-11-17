import dni
import importlib
importlib.reload(dni)
import ray

class AbsFeatureExtractor():
    def __init__(self):
        pass
    def get_next(self):
        pass
    def have_next(self):
        pass
    def extract(self):
        pass
    
class FeatureExtractor(AbsFeatureExtractor):
    def __init__(self, accessmethod, intermediate_function, list_of_featurefunctions):
        self.intermediate = intermediate_function
        self.features = list_of_featurefunctions
        self.c = accessmethod
    def get_next(self):
        if not self.have_next():
            return None    
        input = ray.get(self.c.get_next.remote())    
        return self.extract(input)
    def have_next(self):
        return ray.get(self.c.have_next.remote())
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
        super(self, accessmethod, featurefunction, [lambda a : a])
        
class FeatureFactoryExtractor(FeatureExtractor):
    # featurefactory receives the intermediate data, record and features name, outputs corresponding feature
    def __init__(self, accessmethod, intermediatefunction, featurefactory, names):
        # here we use partial function trick https://www.geeksforgeeks.org/partial-functions-python/
        super(self, accessmethod, intermediatefunction,[featurefactory(intermediate,record,name) for name in names])
    
#dynamicaaly build physical activation extractor based on logical activation extractor
def build_physical_feature_ext(LogicalFeatureExt):
    @ray.remote
    class PhysicalFeatureExt(LogicalFeatureExt):
        def get_next(self):
            if not self.have_next():
                return None
            input_table = ray.get(self.c.get_next.remote())
            f_table = dni.tool.FeatureTable()
            # read the whole batch of inputs
            for num, input in input_table.itr():
                f_table.add_table(self.extract(input))    
            return f_table
        def have_next(self):
            return super().have_next()
    return PhysicalFeatureExt
        