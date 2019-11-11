class Abs_Metric():
    def __init__(self):
        pass
    def extract(self):
        pass
    def get_class(self):
        return "Abs_Metric"
    def get_name(self):
        return self.name

# given two inputs, return their metric
class Exhaustive_Metric(Abs_Metric):
    def __init__(self, metric):
        self.metric = metric
        self.name = ""
    def extract(self,input1,input2):
        return self.metric(input1,input2)
    def get_class(self):
        return "Exhaustive_Metric"

# given two array of inputs, concatenate them, and return metric
class Batch_Exhaustive_Metric(Exhaustive_Metric):
    def __init__(self, metric):
        self.metric = metric
    def extract(self,input1,input2):
        import numpy as np
        while input1.ndim != 1:
            input1 = np.concatenate(input1)
        while input2.ndim != 1:
            input2 = np.concatenate(input2)
        return self.metric(input1,input2)
    def get_class(self):
        return "Batch_Exhaustive_metric"

# batch mode exhaustive correlation
class Batch_Exhaustive_Correlation(Batch_Exhaustive_Metric):
    def __init__(self):
        import numpy as np
        self.metric = np.corrcoef
    # input should from one layer one unit
    def extract(self,input1,input2):
        import numpy as np
        while input1.ndim != 1:
            input1 = np.concatenate(input1)
        while input2.ndim != 1:
            input2 = np.concatenate(input2)
        return self.metric(input1,input2)[1,0]
    def get_class(self):
        return "Batch_Exhaustive_metric"
    def get_name(self):
        return "AbsCorrelation"
    
# incremental correlation
class Incremental_Correlation(Abs_Metric):
    def __init__(self):
        self.sumXsquare = 0
        self.sumYsquare = 0
        self.sumX = 0
        self.sumY = 0
        self.sumXY = 0
        self.n = 0
    # input should from one layer one unit
    def increment(self,input1,input2):
        for i in range(len(input1)):
            self.sumXsquare += input1[i]**2
            self.sumYsquare += input2[i]**2
            self.sumX += input1[i]
            self.sumY += input2[i]
            self.sumXY += input1[i]*input2[i]
            self.n += 1
    def extract(self):
        return (self.sumXY - self.sumX*self.sumY/self.n)/(((self.sumXsquare - self.sumX**2/self.n)**(1/2))*((self.sumYsquare - self.sumY**2/self.n)**(1/2)))
    def get_class(self):
        return "Incremental_Correlation"
    def get_name(self):
        return "Correlation"
    
# batch mode exhaustive correlation
class Batch_Incremental_Comprehensive_Correlation():
    def __init__(self, model, layer, unit, feature):
        import numpy as np
        self.sumXsquare = np.zeros(shape=(model, layer, unit, feature))
        self.sumYsquare = np.zeros(shape=(model, layer, unit, feature))
        self.sumX = np.zeros(shape=(model, layer, unit, feature))
        self.sumY = np.zeros(shape=(model, layer, unit, feature))
        self.sumXY = np.zeros(shape=(model, layer, unit, feature))
        self.n = np.full((model, layer, unit, feature),1e-17)
    # input should from one layer one unit
    def increment(self,input1,input2,model, layer, unit, feature):
        import numpy as np
        while input1.ndim != 1:
            input1 = np.concatenate(input1)
        while input2.ndim != 1:
            input2 = np.concatenate(input2)
        for i in range(len(input1)):
            self.sumXsquare[model, layer, unit, feature] += input1[i]**2
            self.sumYsquare[model, layer, unit, feature] += input2[i]**2
            self.sumX[model, layer, unit, feature] += input1[i]
            self.sumY[model, layer, unit, feature] += input2[i]
            self.sumXY[model, layer, unit, feature] += input1[i]*input2[i]
            self.n += 1
    def extract(self,model, layer, unit, feature):
        return (self.sumXY[model, layer, unit, feature] - self.sumX[model, layer, unit, feature]*self.sumY[model, layer, unit, feature]/\
                self.n[model, layer, unit, feature])/(((self.sumXsquare[model, layer, unit, feature] - self.sumX[model, layer, unit, feature]\
                **2/self.n[model, layer, unit, feature])**(1/2))*((self.sumYsquare[model, layer, unit, feature]\
                - self.sumY[model, layer, unit, feature]**2/self.n[model, layer, unit, feature])**(1/2))+1e-17)
    def get_class(self):
        return "Batch_Incremental_Comprehensive_Correlation"
    def get_name(self):
        return "Correlation"