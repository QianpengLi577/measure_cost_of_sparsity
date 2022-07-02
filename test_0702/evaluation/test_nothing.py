import numpy as np
import torch
from eval import *

core_size = 100
core_num = 10

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        self.fc1 = nn.Linear(784, cfg_fc[0],)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )


    def forward(self, input, time_window = 20):

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        count_spike = 0
        # 路由方面的查找次数是一致的，因此不额外计算
        # 计算为了查找相应的神经元需要计算的次数，第一是计算1bit相加的次数，第二个是计算nbit相加的次数  这个对应的是体系结构7.1-1
        count_add_1 = 0 # direct index 
        count_add_2 = 0 # step index

        count_add_3 = 0 # direct index           7.1-2
        count_add_4 = 0 # two h-direct index     7.1-2
        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x.float(), h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

            for i in range(batch_size):
                x_ = x[i,:].float()
                index_x_ = np.nonzero(x_)
                if len(index_x_)>0:
                    count_spike+=len(index_x_)
                    for j in range(len(index_x_)):
                        for k in range(core_num):
                            if (core_mem_all[k][index_x_[j]][1]>0):
                                temp = core_mem_all[k][index_x_[j]][0]
                                count_add_1+= np.nonzero(temp)[0][len(np.nonzero(temp)[0])-1]-1
                                count_add_2+= core_mem_all[k][index_x_[j]][1]

                l_ = h1_spike[i,:]
                index_l_ = np.nonzero(l_)
                if len(index_l_)>0:
                    count_spike+=len(index_l_)
                    for j in range(len(index_l_)):
                        for k in range(core_num):
                            if (core_mem_all[k][index_l_[j]+784][1]>0):
                                temp = core_mem_all[k][index_l_[j]+784][0]
                                count_add_1+= np.nonzero(temp)[0][len(np.nonzero(temp)[0])-1]-1
                                count_add_2+= core_mem_all[k][index_l_[j]+784][1]

                # event = np.zeros(1024)
                # event[0:784] = x_
                # event[784:912] = l_
                # for i in range(core_num):
                #     for j in range(core_size):
                #         bitmap = core_index_all_0702[i][j][0]
                #         step_indexx =  core_index_all_0702[i][j][1]
                        

        outputs = h2_sumspike / time_window
        # print('add_direct:',count_add_1)
        # print('add_step:',count_add_2)
        # print('spike',count_spike)
        return outputs,count_add_1,count_add_2,count_spike

snn = torch.load('./checkpoint/model_two_fc_0627_.t7')

weight_all = np.zeros((922,922))
fc1_weight = snn.fc1.weight.detach().numpy()
fc2_weight = snn.fc2.weight.detach().numpy()

for i in range(len(fc1_weight)):
    for j in range(len(fc1_weight[i])):
        weight_all[j,i+784]=fc1_weight[i][j]
for i in range(len(fc2_weight)):
    for j in range(len(fc2_weight[i])):
        weight_all[j+784,i+784+128]=fc2_weight[i][j]

th = 0.12
index1 = np.abs(weight_all) >= th
index2 = np.abs(weight_all) < th
index = weight_all.copy()
index[index2] = 0
index[index1] = 1
weight_all[index2]=0
p=np.count_nonzero(index)/(784*128+128*10)
print('sparsity',p)

index_temp = index.T
index = ((index+index.T)>=1).astype(np.int32)

snn.fc1.weight.data= torch.from_numpy(weight_all[0:784,784:912].T).float()
snn.fc2.weight.data= torch.from_numpy(weight_all[784:912,912:922].T).float()


# core_index 表示每个核里对应的真实神经元id core_num*core_size
# core_mem_all有5个mem，每个mem有922个[bitmap，num],bitmap长度为coresize

# shuffle_ix = np.random.permutation(np.arange(core_num*core_size))
# core_index = shuffle_ix.reshape((core_num,core_size))

core_index = np.linspace(0, int(core_num*core_size)-1, int(core_num*core_size)).reshape((core_num,core_size))
core_index = core_index.astype(np.int32)

core_mem_all=[]
for i in range(core_num):
    core_ = core_index[i,:]
    mem = []
    for j in range(922):
        bitmap = np.zeros(core_size)
        for m in range(len(core_)):
            if (core_[m]<922):
                bitmap[m] = index[j,core_[m]]
            else:
                bitmap[m] = 0
        num = np.count_nonzero(bitmap, axis=None)
        mem.append([bitmap,num])
    core_mem_all.append(mem)

N_drc = 1024
len_step_index=8
core_index_all_0702=[]
for i in range(core_num):
    core_ = core_index[i,:]
    mem = []
    for j in range(core_size):
        bitmap = np.zeros(N_drc)
        step_index = np.zeros(int(len_step_index))
        for m in range(922):
            if (core_[j]<922):
                bitmap[m]=index[m,core_[j]]
        for n in range(len_step_index):
            step_index[n]=np.sum(bitmap[0:(n+1)*int(N_drc/len_step_index)])
        # num = np.count_nonzero(bitmap, axis=None)
        mem.append([bitmap,step_index])
    core_index_all_0702.append(mem)



names = 'two_fc_0627_'
data_path =  './raw/' #todo: input your data path
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])



optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
optimizer = lr_scheduler(optimizer, 1, learning_rate, 40)
criterion = nn.MSELoss()
correct = 0
total = 0
begin_time = time.time()

c1=0
c2=0
cs=0

item = 10000/batch_size
ii=0

for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    optimizer.zero_grad()
    outputs,c1t,c2t,cst = snn(inputs)
    labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
    loss = criterion(outputs.cpu(), labels_)
    _, predicted = outputs.cpu().max(1)
    total += float(targets.size(0))
    correct += float(predicted.eq(targets).sum().item())
    c1+=c1t
    c2+=c2t
    cs+=cst
    ii+=1
    print(str(ii)+'/'+str(item))
end_time = time.time()
acc = 100. * float(correct) / float(total)
print('Test Acc: %.5f' % acc)
print('Time: %.5f' % (end_time - begin_time))

print('th',th)
print('sparsity',p)
print('add_direct per neuron per neuron per timestep:',c1/20/10000/(912))
print('add_step per neuron per timestep',c2/20/10000/(912))
print('spike per neuron per timestep',cs/20/10000/(912))

print('nothing')