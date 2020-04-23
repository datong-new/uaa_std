import torch
import os
import cv2



def visual_feature(img, feature, layer):
    mean = feature.mean()
    std = feature.std()
    feature = (feature-mean)/std
    feature = feature.abs()
    sort, indices = torch.sort(feature.view(feature.shape[0]*feature.shape[1]*feature.shape[2]))
    thred = sort[-int(feature.shape[1]*feature.shape[2]*0.1)]
    #target = mean.repeat(feature.shape[2], feature.shape[1], 1)
    #target = target.transpose(0,2)
    feature[feature<thred] = 0

    feature = feature.squeeze(0)
    f = feature.clone()
    feature = feature.expand(3, feature.shape[0], feature.shape[1])
    heat = feature.transpose(2,0).transpose(1,0).cpu().detach().numpy()
#    cv2.imwrite("feature_{}.jpg".format(layer), 0.3*heat*255 + 0.5*img)
    return f

def visual_feature_1(img, feature, layer):
    mean = feature.mean()
    std = feature.abs()
    feature = (feature-mean)/std
    feature = feature.abs()
    #target = mean.repeat(feature.shape[2], feature.shape[1], 1)
    #target = target.transpose(0,2)
    #feature[feature<thred] = 0

    feature = feature.squeeze(0)
    f = feature.clone()
    feature = feature.expand(3, feature.shape[0], feature.shape[1])
    heat = feature.transpose(2,0).transpose(1,0).cpu().detach().numpy()
    cv2.imwrite("feature_{}.jpg".format(layer), 0.3*heat*255 + 0.5*img)
    return f

def visual_feature_2(img, features, img_name):
    features = features.squeeze(0)
    mean = features.view(features.shape[0], -1).mean(1)
    mean = mean.unsqueeze(0)
    h, w = features.shape[1], features.shape[2]
    dot = mean.mm(features.view(features.shape[0], -1))
    dot = dot.view(h, w)
    print("dot shape", dot.shape)
    
    l1 = torch.sqrt((features.view(features.shape[0], -1)**2).sum(0))
    l1 = l1.view(h,w)
    l2 = torch.sqrt((mean**2).sum())
    heat = dot/(l1*l2)
    sort, indices = torch.sort(heat.view(heat.shape[0]*heat.shape[1]))
    thred = sort[int(heat.shape[0]*heat.shape[1]*0.05)]
    heat[heat>thred] = 0
    heat[heat!=0]=1
#    feature = heat
#    feature = feature.expand(3, feature.shape[0], feature.shape[1])
#    heat = feature.transpose(2,0).transpose(1,0).cpu().detach().numpy()
    cv2.imwrite("angle_heat{}.jpg".format(img_name), heat.cpu().detach().numpy()*255)
    cv2.imwrite("angle_heat{}_o.jpg".format(img_name), img)

    
    print("heat shape", heat.shape)
    

"""
f = feature.squeeze(0)
f = f.abs()
f = f.sum(0)
print(f.shape)
"""


img_dir = 'img_dir'
files = os.listdir(img_dir)
for i, img_name in enumerate(files):
    try:
        print(img_name)
        feature = torch.load("./features/{}.jpg.pt".format(img_name.split('.')[0]))
        img = cv2.imread("{}/{}.jpg".format(img_dir, img_name.split('.')[0]))
        img = cv2.resize(img, (feature.shape[3], feature.shape[2]), interpolation=cv2.INTER_LINEAR)
        visual_feature_2(img, feature, img_name)
    except Exception:
        print("except: ", img_name)
    continue
    print("img shape", img.shape)
    print("feature shape", feature.shape)
    
    for i in range(0, feature.shape[1]):
        if i==0: f = visual_feature(img, feature[0,i:i+1,:,:], i)
        else: f += visual_feature(img, feature[0,i:i+1,:,:], i)
    
    
    #mean = f.mean()
    #std = f.std()
    #f = (f-mean)/std
    sort, indices = torch.sort(f.view(f.shape[0]*f.shape[1]))
    thred = sort[-int(f.shape[0]*f.shape[1]*0.1)]
    f[f<thred]=0
    f = f.expand(3, f.shape[0], f.shape[1])
    heat = f.transpose(2,0).transpose(1,0).cpu().detach().numpy()
    cv2.imwrite("./feature_imgs_1/{}_feature.jpg".format(img_name), 0.3*heat*255 + 0.5*img)
    cv2.imwrite("./feature_imgs_1/{}".format(img_name), img)




    
