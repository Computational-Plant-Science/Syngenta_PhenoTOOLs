"""
Version: 1.5

Summary: Plant image traits computation pipeline

Author: suxing liu

Author-email: suxingliu@gmail.com

USAGE:

python pipeline.py 

parameter list:

 ap.add_argument("-p", "--path", required = True, help = "path to image file")
 ap.add_argument("-ft", "--filetype", required=True,    help="Image filetype")


"""

import subprocess, os
import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from fil_finder import FilFinder2D
import astropy.units as u


if __name__ == '__main__':
    
  
    skeleton = cv2.imread("/home/suxing/example/XmviQ.png", 0) #in numpy array format

    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.preprocess_image(flatten_percent=85)
    fil.create_mask(border_masking=True, verbose=False,
    use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

    # Show the longest path
    plt.imshow(fil.skeleton, cmap='gray')
    plt.contour(fil.skeleton_longpath, colors='r')
    plt.axis('off')
    plt.show()
    
    
