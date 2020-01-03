import dni
import importlib
importlib.reload(dni)
import ray
import numpy as np
class AbsActivationExtractor(dni.baseops.UnaryOp):
    def get_next(self):
        raise Exception("Not implemented")
    def has_next(self):
        raise Exception("Not implemented")
    def predict(self):
        raise Exception("Not implemented")
    
class LocalActivationExt(AbsActivationExtractor):
    def __init__(self, accessmethod, modelname, layer,unit):
        super().__init__(accessmethod)
        from keras.models import Model, load_model
        newmodel = load_model(modelname)
        newmodel._make_predict_function()
        outputs = [newmodel.layers[l].output for l in layer]
        self.model = Model(inputs = newmodel.input, outputs = outputs)
        self.unit = unit
    def get_next(self):
        if not self.has_next():
            return None
        input = ray.get(self.c.get_next.remote())
        pred = self.predict(self,input)
        return dni.tool.ActivationTable(pred)
    def has_next(self):
        return ray.get(self.c.has_next.remote())
    def predict(self,input):
        input = self.preprocess(input)
        pred = self.model.predict(input)
        if not isinstance(pred, list):
            pred = [pred]
        for i in range(len(pred)):
            pred[i] = pred[i][...,self.unit]
        return pred
    def preprocess(self, input):
        pass
    
    
#dynamicaaly build physical activation extractor based on logical activation extractor
def build_physical_activation_ext(LogicalActivationExt):
    @ray.remote
    class PhysicalActivationExt(LogicalActivationExt):
        def get_next(self):
            if not self.has_next():
                return None
            input_table = ray.get(self.c.get_next.remote())
            # read the whole batch of inputs
            data = None
            for num, input in input_table.itr():
                new_data = dni.tool.activations_list_to_array(self.predict(input))
                if data is None:
                    data = new_data
                else: 
                    data = np.vstack((data,new_data))
            df = dni.tool.DniDataFrame(data)
            return df
        def has_next(self):
            return super().has_next()
    return PhysicalActivationExt