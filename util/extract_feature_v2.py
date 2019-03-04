# Helper function for extracting features from pre-trained models
import time

import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from backbone.model_irse import IR_50


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img, backbone, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = True):
    # pre-requisites

    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image
    ccropped = ccropped[...,::-1] # BGR to RGB

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped)

    backbone.to(device)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        if tta:
            emb_batch = backbone(ccropped.to(device)).cpu() + backbone(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(backbone(ccropped.to(device)).cpu())
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")

    return features

if __name__ == '__main__':
    path=r'd:/data\photo/'
    backbone_resume_root='../model/backbone_ir50_asia.pth'
    input_size=[112, 112]
    backbone=IR_50(input_size)

    backbone.load_state_dict(torch.load(backbone_resume_root))

    # load backbone from a checkpoint
    print("Loading Backbone Checkpoint '{}'".format(backbone_resume_root))

    files= os.listdir(path)
    for file in files:
        if file.endswith(".jpg"):
            img=cv2.imread(path+file)
            img=cv2.resize(img,(112,112))
            start=time.time()
            features=extract_feature(img,backbone)
            print('time',time.time()-start,features.size())
