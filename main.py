from __future__ import division
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from discriminator import build_discriminator
import scipy.stats as st
import argparse
from psnr_ssim import * #my code

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default="7", help="GPU id for training")
parser.add_argument("--task", default="./pre-trained/", help="path to folder containing the model")
parser.add_argument("--data", default="/media/sharesto/data/SM/data/", help="path to synthetic dataset")
parser.add_argument("--save_model_freq", default=3, type=int, help="frequency to save model")
parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
parser.add_argument("--is_training", default=0, help="training or testing")
parser.add_argument("--continue_training", default=False, action="store_true", help="search for checkpoint in the subfolder specified by `task` argument")
ARGS = parser.parse_args()

gpu_id = ARGS.gpu_id
task=ARGS.task
is_training=ARGS.is_training
continue_training=ARGS.continue_training
hyper=ARGS.is_hyper==1

if is_training:
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
else:
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id

print('CUDA_VISIBLE_DEVICES', gpu_id)

print('is_training:', is_training)
EPS = 1e-12
channel = 64 # number of feature channels to build the model, set to 64
train_syn_root=ARGS.data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool2d(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def relu(x):
    return tf.maximum(0.0, x)


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0] // 2, shape[1] // 2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)

    return _initializer


def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0 * x + w1 * slim.batch_norm(x)


vgg_path = scipy.io.loadmat('./VGG_Model/imagenet-vgg-verydeep-19.mat')

print("[i] Loaded pre-trained vgg19 parameters")

# build VGG19 to load pre-trained parameters
def build_vgg19(input, reuse=False):
    with tf.compat.v1.variable_scope("vgg19"):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        net = {}
        vgg_layers = vgg_path['layers'][0]
        net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        return net


def build_reconnet(input): #BTnet, here in this code it is named as reconstruction net
    if hyper:
        print("[i] Reconnet: Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net

# our reflection removal model
def build(input):
    if hyper:
        print("[i] Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.compat.v1.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3*2,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net

def compute_l1_loss(input, output):
    return tf.reduce_mean(tf.abs(input-output))

def compute_percep_loss(input, output, reuse=False):
    vgg_real=build_vgg19(output*255.0,reuse=reuse)
    vgg_fake=build_vgg19(input*255.0,reuse=True)
    p0=compute_l1_loss(vgg_real['input'],vgg_fake['input'])
    p1=compute_l1_loss(vgg_real['conv1_2'],vgg_fake['conv1_2'])/2.6
    p2=compute_l1_loss(vgg_real['conv2_2'],vgg_fake['conv2_2'])/4.8
    p3=compute_l1_loss(vgg_real['conv3_2'],vgg_fake['conv3_2'])/3.7
    p4=compute_l1_loss(vgg_real['conv4_2'],vgg_fake['conv4_2'])/5.6
    p5=compute_l1_loss(vgg_real['conv5_2'],vgg_fake['conv5_2'])*10/1.5
    return p0+p1+p2+p3+p4+p5

def compute_gradient(img):
    gradx=img[:,1:,:,:]-img[:,:-1,:,:]
    grady=img[:,:,1:,:]-img[:,:,:-1,:]
    return gradx,grady



def prepare_data_test(test_path):
    input_names=[]
    for dirname in test_path:
        for _, _, fnames in sorted(os.walk(dirname)):
            for fname in fnames:
                if is_image_file(fname):
                    input_names.append(os.path.join(dirname, fname))
    return input_names

#--------------------reflection removal network ---------------------------------------------------------------------------------------------

input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
targetgR = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
targetT =tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
targetR = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])

# build the model
network = build(input)
transmission_layer, reflection_layerb4 = tf.split(network, num_or_size_splits=2, axis=3) #split network output to T, gR (reflection_layerb4 == glass reflected R, before applying BTnet)

# Perceptual Loss b4 (perceptual loss of gR, before applying BTnet)
loss_percep_rb4 = compute_percep_loss(reflection_layerb4, targetgR, reuse=True)

# L1 loss on reflection image b4
loss_l1_rb4 = compute_l1_loss(reflection_layerb4, targetgR)  # temp!! activate this line for real training SM

lossb4 = loss_l1_rb4 + loss_percep_rb4 * 0.2

# set up the model and define the graph : BTnet
with tf.compat.v1.variable_scope('R_reconnet'): #R reconstruction== BT-net
    inputgR = reflection_layerb4
    # build the model
    reflection_layer = build_reconnet(inputgR)
    reflection_layer = tf.identity(reflection_layer, name="reflection_layer")

# Perceptual Loss
loss_percep_t = compute_percep_loss(transmission_layer, targetT)
loss_percep_r = compute_percep_loss(reflection_layer, targetR, reuse=True)
loss_percep = loss_percep_t + loss_percep_r

# Adversarial Loss
with tf.variable_scope("discriminator"):
    predict_real, pred_real_dict = build_discriminator(input, targetT)
with tf.variable_scope("discriminator", reuse=True):
    predict_fake, pred_fake_dict = build_discriminator(input, transmission_layer)

d_loss = (tf.reduce_mean(-(tf.math.log(predict_real + EPS) + tf.math.log(1 - predict_fake + EPS)))) * 0.5
g_loss = tf.reduce_mean(-tf.math.log(predict_fake + EPS))

loss_l1_r = compute_l1_loss(reflection_layer, targetR)
loss_l1_t = compute_l1_loss(transmission_layer, targetT)

# lossb4 == a priori loss in the paper
loss = loss_l1_r + loss_percep * 0.2 + loss_l1_t+ lossb4

train_vars = tf.compat.v1.trainable_variables()
d_vars = [var for var in train_vars if 'discriminator' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]

g_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss * 100 + g_loss,
                                                              var_list=g_vars)  # optimizer for the generator
d_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001).minimize(d_loss,
                                                              var_list=d_vars)  # optimizer for the discriminator

#for evaluation
def prepare_data_test(test_path, inputN):
    input_names = []
    for fn in os.listdir(test_path):
        if is_image_file(fn) and inputN in fn:
            input_names.append(os.path.join(test_path,fn))
    return input_names


SPnet_path = task
print('task', task)
savedir = './Results/' # for testing :SAVE_directory

ckpt = tf.train.get_checkpoint_state(SPnet_path)
saver=tf.compat.v1.train.Saver(max_to_keep=110)
# Session starts-------------------------------------------------
sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
if(is_training and not ckpt):
    print('this code for testing, not for training')


else:
    if(not ckpt):
        print('This is testing but no checkpoint. The program will be terminated')
        exit()
    ckpt = tf.train.get_checkpoint_state(SPnet_path)
    saver_restore = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    print('Separation net loaded' + ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)

maxepoch = 140
if is_training:

    print('This code is for testing.')

# To test the model on images with reflection
else:
    print('The testing part is executing')

    modelN = SPnet_path.rsplit('/',3)[1] #model name
    epochN = SPnet_path.rsplit('/',3)[2] #epoch N
    outputdir = os.path.join(savedir,modelN, epochN) #output directory

    #for test images---------------------------------------------------------------------------------------------
    input_path = './test_imgs/blended/'
    input_names =  prepare_data_test(input_path, '.jpg')

    for input_path in input_names:
        testind = os.path.splitext(os.path.basename(input_path))[0] #filename correct for directory seprated as well
        testsetN = input_path.rsplit('/',3)[1] #testset name

        if not os.path.isfile(input_path):
            continue

        img=cv2.imread(input_path)
        input_image=np.expand_dims(np.float32(img), axis=0)/255.0
        st=time.time()
        fetch_list = [transmission_layer, reflection_layerb4, reflection_layer]
        output_image_t, output_image_rb4, output_image_r=sess.run(fetch_list,feed_dict={input:input_image})
        print("Test time %.3f for image %s"%(time.time()-st, input_path))

        output_image_t=np.minimum(np.maximum(output_image_t,0.0),1.0)*255.0
        output_image_rb4=np.minimum(np.maximum(output_image_rb4,0.0),1.0)*255.0
        output_image_r=np.minimum(np.maximum(output_image_r,0.0),1.0)*255.0

        #saving the images
        if not os.path.isdir(os.path.join(outputdir,testsetN)):
            os.makedirs(os.path.join(outputdir,testsetN))
            print('dir is created: %s' %os.path.join(outputdir,testsetN))
        outputdir_fin = os.path.join(outputdir,testsetN)

        cv2.imwrite("%s/%s_input.jpg"%(outputdir_fin,testind),img)
        cv2.imwrite("%s/%s_pred_T.jpg"%(outputdir_fin,testind),np.uint8(output_image_t[0,:,:,0:3])) # output front scene (Transmission)
        cv2.imwrite("%s/%s_pred_gR.jpg"%(outputdir_fin,testind),np.uint8(output_image_rb4[0,:,:,0:3])) # output reflected back scene (Glass reflected reflection)
        cv2.imwrite("%s/%s_pred_R.jpg"%(outputdir_fin,testind),np.uint8(output_image_r[0,:,:,0:3])) # output back scene (Reflection)

    # 'quality_assess_SM3' --> get the numerical numbers if transmission layers are available.----------
    # Comment out the following code, if transmission_layers are not available.
    gtdir = './test_imgs/transmission_layer/'
    quality_assess_SM3(outputdir_fin, gtdir, '_pred_T', '')
    # --------------------------------------------------------------------------------------------------

sess.close()

