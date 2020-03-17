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
    ray.shutdown()
    ray.init(address=clusterid)
    registry = dni.udf.UDFRegistry.registry()
    for index1, elem1 in enumerate(FeaturesFunctions):
        for index2, elem2 in enumerate(elem1):
            if isinstance(elem2, list):
                for index3, elem3 in enumerate(elem2):
                    if type(elem3) is str:
                        FeaturesFunctions[index1][index2][index3] = registry[elem3]
            else:
                if type(elem2) is str:
                    FeaturesFunctions[index1][index2] = registry[elem2]
    nummodel = len(Models)
    layers = []
    units = []
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
    file_name = AccessMethod[0][1]
    batch_size = 2


    phy = dni.access.build_physical_scanner(dni.access.LocalScanner)
    ps1 = phy.remote(file_name,batch_size)
    phy_f_act = dni.feature.build_physical_feature_ext(dni.feature.FeatureExtractor)
    fxt = phy_f_act.remote(ps1,FeaturesFunctions[0][1],FeaturesFunctions[0][0])
    ps2 = phy.remote(file_name,batch_size)
    phy_m_act = dni.model.build_physical_activation_ext(ActivationExt)
    m = phy_m_act.remote(ps2,Models[0],layers[0],units[0])
    corr_metric = dni.metric.IncrementalCorrelation(fxt,m,nummodel,maxlayer,maxunits,numfeature)
    corr_metric.open_itr()
    from prettytable import PrettyTable
    t = PrettyTable(['Model','Metric','Feature','Neuron','Score'])
    #for each model
    for i in range(len(Models)):
        # for each layer
        for j in range(len(layers[i])):
            # for each unit
            for k in range(len(units[i])):
                # for each feature
                for l in range(2):
                    t.add_row([Models[i],MetricName[j],FeatureNames[l],"("+str(layers[i][j])+","+str(units[i][k])+")",corr_metric.extract(0,j,k,l)])               
    print(t)