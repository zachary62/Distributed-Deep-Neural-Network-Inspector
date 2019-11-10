import ray
import numpy as np
import dni
import importlib
importlib.reload(dni)

def execute(clusterid ,
            AccessMethod,
            ActivationExt, Neuron, Models,
            FeaturesFunctions , FeatureNames, FeatureExtractor,
            MetricExtractor, MetricName):
    layers = []
    units = []
    ray.shutdown()
    ray.init(address=clusterid)
    for i in range(len(Neuron)):
        layers.append(Neuron[i][0])
        units.append(Neuron[i][1])
    ModelActor = ActivationExt.remote(Models[0],layers[0])
    FeatActor = [None]*len(FeatureNames)
    for i in range(len(FeatureNames)):
        FeatActor[i] = FeatureExtractor[i].remote(FeaturesFunctions[i],FeatureNames[i])
    ScannerActor = AccessMethod.remote()
    act = []
    feature = [[] for i in range(len(FeatureNames))]
    while ray.get(ScannerActor.HasMore.remote()):
        data = ScannerActor.Next.remote()
        Ext = ModelActor.extract.remote(data,units[0])
        act.append(ray.get(Ext))
        for i in range(len(FeatureNames)):
            feature[i].append(ray.get(FeatActor[i].extract.remote(data)))

    from prettytable import PrettyTable
    t = PrettyTable(['Model','Metric','Feature','Neuron','Score'])
    
    
    input1 = np.array(act)
    input2 = np.array(feature)
    
    mmm = MetricExtractor[0].remote()
    
    #for each model
    for i in range(len(Models)):
        # for each layer
        for j in range(len(layers[i])):
            # for each unit
            for k in range(len(units[i])):
                # for each feature
                for l in range(len(FeatureNames)):
                    m = mmm.extract.remote(input1[:,j,:,:,k],input2[l])
                    t.add_row([Models[i],MetricName[j],FeatureNames[l],"("+str(layers[i][j])+","+str(units[i][k])+")",ray.get(m)])

    print(t)
        
    
    