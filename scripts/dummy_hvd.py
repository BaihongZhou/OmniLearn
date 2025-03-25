# to support the env without horovod
class dummy_hvd:
    def __init__(self):
        pass
    def rank(self):
        return 0
    def size(self):
        return 1
    def init(self):
        return
    def local_rank(self):
        return 0
    def DistributedOptimizer(self,o):
        return o
    def allgather(self,x):
        return x
    def allreduce(self,x):
        return x
hvd=dummy_hvd()