import matplotlib.pyplot as plt
h10=[]
h20=[]
h3=[]
h1=[]
mr=[]
mrr=[]
epoch=[]

with open("/Users/kaiwang/Desktop/v6-query-r-f-metric.txt",'r') as f:

    lines=f.readlines()
    for line in lines:
        h10.append(float(line.replace('\n','').split(',')[0]))
        h20.append(float(line.replace('\n','').split(',')[1]))
        h3.append(float(line.replace('\n','').split(',')[2]))
        h1.append(float(line.replace('\n','').split(',')[3]))
        mr.append(float(line.replace('\n','').split(',')[4]))
        mrr.append(float(line.replace('\n','').split(',')[5]))
        epoch.append(float(line.replace('\n','').split(',')[6]))

plt.figure(figsize=(15, 8))
plt.subplot(231)
plt.plot(epoch,h10)
# plt.xlim((-5, 5))
# plt.ylim((0, 0.0015))
plt.xlabel('epoch')
plt.ylabel('@10')

plt.subplot(232)
plt.plot(epoch,h20)
# plt.xlim((-5, 5))
# plt.ylim((0, 0.0015))
plt.xlabel('epoch')
plt.ylabel('@20')
plt.subplot(233)

plt.plot(epoch,h3)
# plt.xlim((-5, 5))
# plt.ylim((0, 0.0015))
plt.xlabel('epoch')
plt.ylabel('@3')

plt.subplot(234)
plt.plot(epoch,h1)
# plt.xlim((-5, 5))
# plt.ylim((0, 0.0015))
plt.xlabel('epoch')
plt.ylabel('@1')

plt.subplot(235)
plt.plot(epoch,mr)
# plt.xlim((-5, 5))
plt.ylim((0, 20))
plt.xlabel('epoch')
plt.ylabel('mr')

plt.subplot(236)
plt.plot(epoch,mrr)
# plt.xlim((-5, 5))
# plt.ylim((0, 0.0015))
plt.xlabel('epoch')
plt.ylabel('mrr')

plt.show()