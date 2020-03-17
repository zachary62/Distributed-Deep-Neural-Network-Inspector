import dni
import importlib
importlib.reload(dni)
import ray

class BroadCaster(dni.baseops.UnaryOp):
    def get_next(self):
        # get tables from child and merge all of them
        if self.table == None:
            table = dni.tool.HighDimensionPartitionableTable()
            while ray.get(self.c.has_next.remote())
                table.merge(ray.get(self.feature_child.get_next.remote()))
            self.table = table
        # return tables to parent
        return self.table
    def has_next(self):
        return True

class Partitioner(dni.baseops.UnaryOp):
    def __init__(self, child, partition_number):
        super().__init__(child)
        self.partition_number = partition_number
        self.stack_of_partitions = []
    def get_next(self):
        # get table from child and partition it
        if len(self.stack_of_partitions) == 0:
            self.stack_of_partitions += ray.get(self.c.get_next.remote()).partition(self.partition_number)
        return self.stack_of_partitions.pop()
    def has_next(self):
        return ray.get(self.c.has_next.remote()) or len(self.stack_of_partitions) > 0