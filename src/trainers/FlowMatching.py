from .MTT import MTT, MTTmodel, MTTdata

class FlowMatching(MTT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class FMmodel(MTTmodel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        print('just a test\n')
        front, label = batch
        #self.data_fetch_start_time = time.time() - self.data_fetch_start_time
        #self.forward_start_time = time.time()
        pred = self(front)
        #self.lossbackward_start_time = time.time()
        train_loss = F.mse_loss(pred, label)
        self.train_losses.append(train_loss.item())
        return train_loss
    
class FMdata(MTTdata):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
