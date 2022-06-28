from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from model_snn import*
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# names = 'model_snn'
data_path =  './raw/' #todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

snn = torch.load('./checkpoint/model_model_snn.t7')

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