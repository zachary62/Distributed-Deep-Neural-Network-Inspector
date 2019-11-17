import dni
import importlib
importlib.reload(dni)
import ray

class AbsActivationExtractor():
    def __init__(self):
        pass
    def get_next(self):
        pass
    def have_next(self):
        pass
    def predict(self):
        pass
    
class LocalActivationExt(AbsActivationExtractor):
    def __init__(self, accessmethod, modelname, layer,unit):
        from keras.models import Model, load_model
        newmodel = load_model(modelname)
        newmodel._make_predict_function()
        outputs = [newmodel.layers[l].output for l in layer]
        self.model = Model(inputs = newmodel.input, outputs = outputs)
        self.unit = unit
        self.c = accessmethod
    def get_next(self):
        if not self.have_next():
            return None
        input = ray.get(self.c.get_next.remote())
        pred = self.predict(self,input)
        return dni.tool.ActivationTable(pred)
    def have_next(self):
        return ray.get(self.c.have_next.remote())
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
            if not self.have_next():
                return None
            input_table = ray.get(self.c.get_next.remote())
            act_table = dni.tool.ActivationTable()
            # read the whole batch of inputs
            for num, input in input_table.itr():
                act_table.add_table(self.predict(input))    
            return act_table
        def have_next(self):
            return super().have_next()
    return PhysicalActivationExt