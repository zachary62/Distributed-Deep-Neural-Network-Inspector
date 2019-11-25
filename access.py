import dni
import importlib
importlib.reload(dni)
import ray

class AccessMethod(dni.baseops.UnaryOp):
    def get_next(self):
        raise Exception("Not implemented")
    def has_next(self):
        raise Exception("Not implemented")

class LocalScanner(AccessMethod):
    def __init__(self, filename):
        self.f = open(filename, 'r')
        self.f.seek(0)
        self.nextline = self.f.readline()
    def get_next(self):
        if not self.has_next():
            return None
        currentline = self.nextline
        self.nextline = self.f.readline()
        return self.post_process(currentline)
    def has_next(self):
        if self.nextline:
            return True
        else:
            return False
    def post_process(self,sttr):
        return sttr.split("\n")[0]

#dynamicaaly build physical scanner based on logical scanner
def build_physical_scanner(LogicalScanner):
    @ray.remote
    class PhysicalScanner(LogicalScanner):
        def __init__(self,file,batch):
            super().__init__(file)
            self.batch = batch  
        def get_next(self):
            batch_inputs = []
            while self.has_next() and len(batch_inputs)<self.batch:
                batch_inputs.append(super().get_next())
            return dni.tool.InputTable(batch_inputs) 
        def has_next(self):
            return super().has_next()
    return PhysicalScanner
