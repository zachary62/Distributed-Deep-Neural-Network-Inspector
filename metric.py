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