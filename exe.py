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

    maxlayer = 0
    maxunits = 0
    for i in range(len(Models)):
        maxlayer = max(maxlayer,len(layers[i]))
        maxunits = max(maxunits,len(units[i]))

    numfeature = 0
    for i in range(len(FeaturesFunctions)):
        if type(FeaturesFunctions[i]) is list and len(FeaturesFunctions[i]) == 2:
            numfeature += len(FeaturesFunctions[i][0])
        else:
            numfeature += 1

    metrics = [None]*len(MetricName)
    for i in range(len(MetricName)):
        if MetricName[i] == "Correlation":
            metrics[i] = dni.metric.Batch_Incremental_Comprehensive_Correlation(len(Models),maxlayer,maxunits,numfeature)

    ModelActor = ActivationExt.remote(Models[0],layers[0])

    FeatActor = [None]*len(FeaturesFunctions)
    for i in range(len(FeaturesFunctions)):
        # if has intermediate result
        if type(FeaturesFunctions[i]) is list and len(FeaturesFunctions[i]) == 2:
            featureclass = ray.remote(dni.feature.CachedFeatureExtractor)
            FeatActor[i] = featureclass.remote(FeaturesFunctions[i][0],FeaturesFunctions[i][1])
        else:
            FeatActor[i] = FeatureExtractor[i].remote(FeaturesFunctions[i],FeatureNames[i])

    ScannerActor = AccessMethod.remote()
    act = []
    feature = [[] for i in range(len(FeatureNames))]
    while ray.get(ScannerActor.HasMore.remote()):
        data = ScannerActor.Next.remote()
        Ext = ModelActor.extract.remote(data,units[0])
        act.append(ray.get(Ext))
        currf = 0
        # for each group of feature functions
        for i in range(len(FeaturesFunctions)):
            if type(FeaturesFunctions[i]) is list and len(FeaturesFunctions[i]) == 2:
                FeatActor[i].setup.remote(data)
                # for each feature functions
                for j in range(len(FeaturesFunctions[i][0])):
                    FeatActor[i].extract.remote(data,1)
                    feature = ray.get(FeatActor[i].extract.remote(data,j))
                    # for each layer
                    for k in range(len(layers[0])):
                        # for each units
                        for l in range(len(units[i])):
                            # for each metric
                            for m in range(len(MetricName)):
                                metrics[i].increment(feature,ray.get(Ext)[k][...,l],0,k,l,currf)
                    currf += 1
            else:
                feature[currf].append(ray.get(FeatActor[i].extract.remote(data)))
                currf += 1

    from prettytable import PrettyTable
    t = PrettyTable(['Model','Metric','Feature','Neuron','Score'])

    #for each model
    for i in range(len(Models)):
        # for each layer
        for j in range(len(layers[i])):
            # for each unit
            for k in range(len(units[i])):
                # for each feature
                for l in range(numfeature):
                    t.add_row([Models[i],MetricName[j],FeatureNames[l],"("+str(layers[i][j])+","+str(units[i][k])+")",metrics[i].extract(0,j,k,l)])

    print(t)

        
    
    