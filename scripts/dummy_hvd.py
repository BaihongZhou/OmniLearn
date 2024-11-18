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
hvd=dummy_hvd()