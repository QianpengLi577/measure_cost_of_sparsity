import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 100 # max epoch
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike


# fc layer
cfg_fc = [10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        self.fc1 = nn.Linear(784, cfg_fc[0],)


    def forward(self, input, time_window = 20):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)


        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x.float(), h1_mem, h1_spike)
            h1_sumspike += h1_spike


        outputs = h1_sumspike / time_window
        return outputs


