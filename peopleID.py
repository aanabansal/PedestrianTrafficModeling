import json
import apollocaffe
import os
import cv2
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.misc import imread
import apollocaffe # Make sure that caffe is on the python path:
from utils.annolist import AnnotationLib as al
from train import load_idl, forward
from utils import load_data_mean, Rect, stitch_rects
blocks = ['top', 'bottom', 'left', 'right', 'block']
# # Count number of people in each set and store in 5 X 10 array
data = [[0 for i in range(5)] for j in range(6)]
print(str(data))
# Build IDL files
for b in blocks:
    for w in range(1,7):
        path = '/home/ubuntu/apollocaffe/reinspect/data/experiments/ex5/' + b + '/m' + str(w)
        idl = open('/home/ubuntu/apollocaffe/reinspect/data/idlFiles/ex5/' + b + str(w) + '.idl', 'w+')
        numImages = 0
        for root, dirs, filenames in os.walk(path):
            for f in filenames:
                if f != 'data.json' and f != '':
                    #print(str("adding: " + "../experiments/"+ b + '/w' + str(w) + '/' + f + "\":;\n"))    
                    idl.write("\"../../experiments/ex5/"+ b + '/m' + str(w) + '/' + f + "\":;\n")
                    numImages = numImages + 1
            idl.seek(-1, os.SEEK_END)
            idl.truncate()
        config = json.load(open("config.json", 'r'))
        config["data"]["test_idl"] = idl.name
        # print("idl =  " + str(config["data"]["test_idl"]))
        apollocaffe.set_random_seed(config["solver"]["random_seed"])
        apollocaffe.set_device(0)
        data_mean = load_data_mean(config["data"]["idl_mean"],
                            config["net"]["img_width"],
                            config["net"]["img_height"], image_scaling=1.0)
       # num_test_images = 500
       #  print("data_mean = " + str(data_mean))

       #  print("config[net] = " + str(config["net"]))
        # Warning: load_idl returns an infinite generator. Calling list() before islice() will hang.
        test_list = list(itertools.islice(
                load_idl(config["data"]["test_idl"], data_mean, config["net"], False),
                0,
                numImages))

        print("initializing net")
        net = apollocaffe.ApolloNet()
        net_config = config["net"]
        net.phase = 'test'
        forward(net, test_list[0], config["net"], True)
        net.load("./data/brainwash_800000.h5")
        print 'b = ' + b
        print 'w = ' + str(w)
        count = 0
        pix_per_w = net_config["img_width"]/net_config["grid_width"]
        pix_per_h = net_config["img_height"]/net_config["grid_height"]
        forward(net, test_list[0], config["net"], True)
        print("len(test_list) = " + str(len(test_list)))
        for i in range(0, len(test_list)):
            c = 0
            #print(str(i))
            inputs = test_list[i]
            bbox_list, conf_list = forward(net, inputs, net_config, True)
            all_rects = [[[] for x in range(net_config["grid_width"])] for y in range(net_config["grid_height"])]
            for n in range(len(bbox_list)):
                for k in range(net_config["grid_height"] * net_config["grid_width"]):
                    y = int(k / net_config["grid_width"])
                    x = int(k % net_config["grid_width"])
                    bbox = bbox_list[n][k]
                    conf = conf_list[n][k,1].flatten()[0]
                    abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
                    abs_cy = pix_per_h/2 + pix_per_h*y+int(bbox[1,0,0])
                    wi = bbox[2,0,0]
                    h = bbox[3,0,0]
                    all_rects[y][x].append(Rect(abs_cx,abs_cy,wi,h,conf))
            acc_rects = stitch_rects(all_rects)
            imCount = 0
            for rect in acc_rects:
                if rect.true_confidence > 0.8:
                    imCount = imCount + 1
            print(inputs["imname"])
            #print("imCount = " + str(imCount))
            count = count + imCount
        print("count = " + str(count))         
        a = blocks.index(b)
        print("w = " + str(w))
        print("a = " + str(a))
        data[w - 1][a] = count
        print(str(data))
print(str(data))
