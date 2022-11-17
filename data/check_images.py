# import os
# path='/Users/kaiwang/Desktop/code/MKGformer-main/pretrain/FB15k-images'
# enitiy_images=os.listdir(path)
# print("总数量:",len(enitiy_images))
# entity_no_images=[]
# for i in enitiy_images:
#     if os.path.isdir(path+'/'+i):
#         nums=len(os.listdir(path+'/'+i))
#         # print(i,nums)
#         if nums==0:
#             entity_no_images.append(i)
# print("没有图片的实体数量:",len(entity_no_images))

entity_path='/Users/kaiwang/Desktop/entity2textlong.txt'
with open(entity_path, 'r') as f:
    print("!"*10,entity_path)
    lines = f.readlines()
    entities = []
    for line in lines:
        entities.append(line.strip().split("\t")[0])
    print('len(entities):',len(entities))