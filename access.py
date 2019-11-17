import dni
import importlib
importlib.reload(dni)
import ray

class AccessMethod():
    def __init__(self):
        pass
    def get_next(self):
        pass
    def have_next(self):
        pass

class LocalScanner(AccessMethod):
    def __init__(self, filename):
        self.f = open(filename, 'r')
        self.f.seek(0)
        self.nextline = self.f.readline()
    def get_next(self):
        if not self.have_next():
            return None
        currentline = self.nextline
        self.nextline = self.f.readline()
        return self.post_process(currentline)
    def have_next(self):
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
            while self.have_next() and len(batch_inputs)<self.batch:
                batch_inputs.append(super().get_next())
            return dni.tool.InputTable(batch_inputs) 
        def have_next(self):
            return super().have_next()
    return PhysicalScanner