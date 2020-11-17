import os
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import cv2
from model import colorNet
from utils import *
import numpy as np
import matplotlib.pyplot as plt

#load the data
data_path = "./test_data/"
have_cuda = torch.cuda.is_available() #will be true if GPU is detected

test_set = testImageFolder(data_path)
test_set_size = len(test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=1)




def test():
    
    color_model = colorNet()
    #loading the pretrained weights
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
    
    if have_cuda:
        color_model.cuda()
    
    color_model.eval()
    i = 0
    for data, _ in test_loader:
        img_orig = data[0].unsqueeze(1).float()
        file_name = './black_white/' + str(i) + '.jpg'
        for img in img_orig:
            sample = img.squeeze().numpy()
            sample = sample.astype(np.float64)
            plt.imsave(file_name, sample, cmap='gray')
        
        w = img_orig.size()[2]
        h = img_orig.size()[3]
        scale_img = data[1].unsqueeze(1).float()
        if have_cuda:
            img_orig, scale_img = img_orig.cuda(), scale_img.cuda()
            color_model = color_model.cuda()
        
        img_orig, scale_img = Variable(img_orig, volatile=True), Variable(scale_img)
        _, output = color_model(img_orig, scale_img)
        color_img = torch.cat((img_orig, output[:, :, 0:w, 0:h]), 1)
        color_img = color_img.data.cpu().numpy().transpose((0,2,3,1))
        
        #scale factor to normalize the output of the tanh function
        #scale_param = (255 - 128)
        for img in color_img:
            print("### BEFORE SCALING ###")
            print(img)
            #img[:,:,0:1] = img[:, :, 0:1] * 255
            img[:,:,1:3] = img[:, :, 1:3] * 255 - 128 #to rescale the distribution to get the original image
            img = np.array(img, dtype=np.uint8)
            print("### AFTER SCALING ###")
            print(img)
            img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
            color_file = './color/' + str(i) + '.jpg'
            plt.imsave(color_file, img)
            i += 1
          

test()
  
