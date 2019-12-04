import numpy as np
import os
from scipy.spatial import KDTree
from sklearn.preprocessing import normalize
#from laspy_utils import  RasterizeLiDAR, LiDAR
from scipy.interpolate import griddata
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from PIL import Image
import torch

if __name__ == '__main__':
    ground_id= 2

    #print(filefilst)

    # prepare trained model
    opt = TestOptions().parse()  # get test options
    print("    set up ")
    print(opt)
    child_dir = opt.dataroot  #+ "\""   #D:/ALS/TM/manual/LAS_AGL1500/LAS/"
    filefilst = os.listdir(child_dir)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.preprocess = 'none'
    opt.model = 'test'
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    # test with eval mode. This only affects layers like batchnorm and dropout.

    model.eval()
    h = 256  # 512
    w = 256  # 512
    device = torch.device("cuda")
    mode = "cuda"
    model_ = torch.jit.trace(model.netG, torch.rand(1, 3, h, w).to(device))
    model_.save("Net_h{}_w{}_{}.pt".format(h, w, mode))
    print("DepthNet_h{}_w{}_{}.pt is exported".format(h, w, mode))




