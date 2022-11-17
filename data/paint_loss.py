import matplotlib.pyplot as plt
t=[]
loss=[]
with open("/Users/kaiwang/Desktop/v6-query-r-f-loss.txt",'r') as f:
    lines=f.readlines()

    for line in lines:
        t.append(int(line.replace('\n','').split(',')[0]))
        loss.append(float(line.replace('\n','').split(',')[1]))
        # print(t,loss)

plt.plot(t,loss)
# plt.xlim((-5, 5))
plt.ylim((0, 0.001))

plt.xlabel('step')
plt.ylabel('loss')
plt.show()