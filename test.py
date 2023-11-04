# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 22:19:21 2022

@author: 80594
"""

import torch
from torch.utils.data import DataLoader
from testloader import GTSRB_Test_Loader
from evaluation import evaluate
from net import Net


if __name__ == '__main__':
    torch.manual_seed(118)
    testloader = DataLoader(GTSRB_Test_Loader(TEST_PATH='./GTSRB_Final_Test_Images/Final_Test/Images/', TEST_GT_PATH='./GTSRB_Test_GT.csv'), batch_size=50, shuffle=True, num_workers=8)

    # import your trained model 
    PATH = './net.pth'
    net = Net(43)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    # can test in cpu
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    testing_accuracy = evaluate(net, testloader)
    print('testing finished, accuracy: {:.3f}'.format(testing_accuracy))