class UDFRegistry(object):
    """
    Global singleton object for managing registered UDFs
    """
    _registry = None
    def __init__(self):
        self.udfs = {}
    @staticmethod
    def registry():
        if not UDFRegistry._registry:
            UDFRegistry._registry = UDFRegistry()
        return UDFRegistry._registry  
    def add(self, udf, name):
        if name in self.udfs:
            raise Exception("An UDF with same name already exists %s" % name)
        self.udfs[name] = udf
    def __getitem__(self, name):
        if name in self.udfs:
            return self.udfs[name]
        raise Exception("Could not find UDF named %s" % name)