from .MTT import MTT, MTTmodel, MTTdata

class FlowMatching(MTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FMmodel(MTTmodel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
class FMdata(MTTdata):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
