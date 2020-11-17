import os, argparse
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from utils import *
from model import colorNet

# Parse arguments and prepare program
parser = argparse.ArgumentParser(description='Training Network')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='size of mini-batch (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='learning rate at start of training')
args = parser.parse_args()
# print("Path to data:" + str(args.data))
# print("Workers:" + str(args.workers))
# print("Epochs:" + str(args.epochs))
# print("Batch size:" + str(args.batch_size))
# print("Learning rate:" + str(args.lr))



original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

#initializing the weights by xavier initialization
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)




have_cuda = torch.cuda.is_available()
epochs = args.epochs

#path to augmented training set
print("Loading images")
data_dir = args.data
train_set = TrainImageFolder(data_dir, original_transform)
print(train_set)
train_set_size = len(train_set)
print("######## TRAIN SET SIZE #############")
print(train_set_size)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
color_model = colorNet()
if os.path.exists('./colornet_params.pkl'):
    color_model.load_state_dict(torch.load('colornet_params.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adam(color_model.parameters())
criterion = torch.nn.MSELoss().cuda() if have_cuda else torch.nn.MSELoss()
#color_model.apply(weights_init) 
print("Parameters set")

def train(epoch):
    color_model.train()
    

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            messagefile = open('./loss_log.txt', 'a')
            #print("for loop e dhukeche")
            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
                
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            #print("Optimizer set koreche")
            class_output, output = color_model(original_img, original_img)
            #print("Model output peye geche")
            # print(output.shape)
            # print(img_ab.shape)
            # exit()
            ems_loss = criterion(img_ab, output) # MSE
            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
            loss = ems_loss + cross_entropy_loss
            lossmsg = 'loss: %.9f\n' % (loss.data)
            messagefile.write(lossmsg)
            ems_loss.backward(retain_graph=True)
            cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data)
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params_tesla.pkl')
            messagefile.close()
            print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data))

    except Exception:
        logfile = open('error_log.txt', 'a')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')

for epoch in range(1, epochs + 1):
    #print("train korche")
    train(epoch)



