class Access_Method():
    pass

class Scan(Access_Method):
    def __init__(self):
        pass
    def Next(self):
        pass
    def HasMore(self):
        pass
    def get_class(self):
        return "Access_Scan"

class Index(Access_Method):
    total = 0
    batch = 0
    def __init__(self):
        pass
    def Lookup(self):
        pass
    def get_class(self):
        return "Access_Index"