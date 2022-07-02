from itertools import count
import numpy as np

sparsity = 0.1
num = 128
length = 4


# print('sparsity:',np.sum(bitmap)/num)

def b1(bitmap, index):
    return int((index+1)/16)+1, index+1-1

def b2(bitmap, map_2, index):
    count_add_1=0
    count_add_2=0
    count_read=0
    if ((index+1)%(num/length)==0):
        count_read = 2
        count_add_1 = 0
        count_add_2 = 0
    else :
        count_read = 2
        count_add_1 = (index+1)%16 - 1
        count_add_2 = 1
    return count_add_1, count_add_2, count_read



iters=100000
l_l=[4,8,16]
n_l=[32,64,128,256,512,1024]
for l in l_l:
    b1_1_alln=[]
    b1_2_alln=[]
    b2_1_alln=[]
    b2_2_alln=[]
    b2_3_alln=[]
    for n in n_l:
        b1_1_all=[]
        b1_2_all=[]
        b2_1_all=[]
        b2_2_all=[]
        b2_3_all=[]
        for m in range(iters):
            bitmap = np.random.binomial(1, p=sparsity, size=n)
            src = np.random.randint(0, high=n)
            bitmap[src]=1
            map_2=np.zeros(l)
            for i in range(l):
                map_2[i]=np.sum(bitmap[0:(i+1)*int(n/l)])

            b1_1 ,b1_2 = b1(bitmap,src)
            b2_1, b2_2, b2_3 = b2(bitmap,map_2,src)
            b1_1_all.append(b1_1)
            b1_2_all.append(b1_2)
            b2_1_all.append(b2_1)
            b2_2_all.append(b2_2)
            b2_3_all.append(b2_3)
        b1_1_alln.append(np.sum(np.array(b1_1_all))/iters)
        b1_2_alln.append(np.sum(np.array(b1_2_all))/iters)
        b2_1_alln.append(np.sum(np.array(b2_1_all))/iters)
        b2_2_alln.append(np.sum(np.array(b2_2_all))/iters)
        b2_3_alln.append(np.sum(np.array(b2_3_all))/iters)
    print('l:',l)
    print('b1')
    print('read:',b1_1_alln)
    print('add:',b1_2_alln)

    print('b2')
    # print('read:',b1_1_alln)
    print('add-1:',b2_1_alln)
    print('add-2:',b2_2_alln)


# print('b1')
# print('read:',np.sum(np.array(b1_1_all))/iters)
# print('add:',np.sum(np.array(b1_2_all))/iters)

# print('b2')
# print('read:',np.sum(np.array(b2_3_all))/iters)
# print('add-1:',np.sum(np.array(b2_1_all))/iters)
# print('add-2:',np.sum(np.array(b2_2_all))/iters)