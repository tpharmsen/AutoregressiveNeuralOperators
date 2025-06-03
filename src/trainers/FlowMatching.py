from MTT import MTT, MTTmodel, MTTdata

class FlowMatching(MTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FMmodel(MTTmodel):
    def forward(self, x):
        # override just this method
        return super().forward(x) + 1  # example custom logic
    
class FMdata(MTTdata):
    def setup(self, stage=None):
        super().setup(stage)
