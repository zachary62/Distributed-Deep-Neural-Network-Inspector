import ray
import importlib
import access
import model
import exe
import feature
import metric
import tool
importlib.reload(access)
importlib.reload(model)
importlib.reload(exe)
importlib.reload(feature)
importlib.reload(metric)
importlib.reload(tool)

def inspect(clusterid ,
            AccessMethod,
            ActivationExt, Neuron, Models,
            FeaturesFunctions , FeatureNames, FeatureExtractor,
            MetricExtractor, MetricName):
    
    optimize()
    
    return exe.execute(clusterid ,
            AccessMethod,
            ActivationExt, Neuron, Models,
            FeaturesFunctions , FeatureNames, FeatureExtractor,
            MetricExtractor, MetricName)
    

def run(f):
    return ray.get(f.remote())
    
def optimize():
    pass
    
class Extractor():
    def __init__(self):
        pass
    def extract(self):
        pass
