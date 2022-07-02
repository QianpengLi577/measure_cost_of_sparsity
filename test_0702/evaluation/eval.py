from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time



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
cfg_fc = [128, 10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

# class SCNN(nn.Module):
#     def __init__(self):
#         super(SCNN, self).__init__()

#         self.fc1 = nn.Linear(784, cfg_fc[0],)
#         self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )


#     def forward(self, input, time_window = 20):

#         h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
#         h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

#         for step in range(time_window): # simulation time steps
#             x = input > torch.rand(input.size(), device=device) # prob. firing
#             x = x.view(batch_size, -1)

#             h1_mem, h1_spike = mem_update(self.fc1, x.float(), h1_mem, h1_spike)
#             h1_sumspike += h1_spike
#             h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
#             h2_sumspike += h2_spike

#         outputs = h2_sumspike / time_window
#         return outputs



# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
if __name__=="__main__":
    names = 'two_fc_0627_'
    data_path =  './raw/' #todo: input your data path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    acc_record = list([])



    snn = torch.load('./checkpoint/model_two_fc_0627_.t7')
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
    optimizer = lr_scheduler(optimizer, 1, learning_rate, 40)
    criterion = nn.MSELoss()
    correct = 0
    total = 0
    begin_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = snn(inputs)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    end_time = time.time()
    acc = 100. * float(correct) / float(total)
    print('Test Acc: %.5f' % acc)
    print('Time: %.5f' % (end_time - begin_time))