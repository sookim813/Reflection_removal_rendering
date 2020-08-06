import numpy as np
import os
import cv2
import math

from skimage.measure import compare_ssim, compare_psnr
from functools import partial


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
def isImagefile(input):
    return any(input.endswith(ext) for ext in IMG_EXTENSIONS)


def quality_assess_SM3(inputdir, gtdir, Xname,GTname):
    imgs = [fn for fn in os.listdir(inputdir) if isImagefile(fn) and Xname in fn]

    fc = open(os.path.join(inputdir,'result.txt'), 'w')

    ssim_list = []
    psnr_list = []

    for fn in imgs:

        GTn = fn.replace(Xname, GTname)
        print('-------------------------------------------------')
        print('GTname:', GTn)
        print('Compared name:', fn)

        inputImg = cv2.imread(os.path.join(inputdir,fn))
        GT = cv2.imread(os.path.join(gtdir,GTn))

        inputImg_gry = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
        GT_gry = cv2.cvtColor(GT, cv2.COLOR_BGR2GRAY)


        p = compare_psnr(inputImg, GT)
        print('p', p)
        psnr_list.append(p)

        s = compare_ssim(inputImg_gry,GT_gry)
        print('s', s)
        ssim_list.append(s)

        fc.write('%s : %.4f %.4f \n' %(GTn, p, s))
    fc.write('Avg. psnr: %f ssim: %f\n' %(np.mean(psnr_list), np.mean(ssim_list)))
    fc.close()

def dropext(fn): return os.path.splitext(fn)[0]
