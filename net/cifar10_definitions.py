#########################################################################
#       Set of networks designed for Cifar-10
#########################################################################

import theano
import theano.tensor as T

import lasagne
import lasagne.layers as L
import lasagne.regularization as R
import lasagne.nonlinearities as NL
import lasagne.objectives as O
import lasagne.init as I

import fwrf.src.numpy_utility as pnu
import fwrf.src.lasagne_utility as plu
from fwrf.src.lasagne_utility import deconv, conv, batch_norm, batch_norm_n, fc_concat, \
    conv_concat, avg, flatten, sigmoid, tanh, winit


npc = 3         # # of channels in image
npx = 32        # # of pixels width/height of images 
ny  = 10        # # of classes

###########################################################
##  CLASSIFICATION 
###########################################################
cls_net_str = '128c4s2_256c4s2_384c3_512c3_1024c1_gp_10f'
def cls_net(_incoming, noise_sigma=0.05, drop_ratio_conv=0.3, drop_ratio_fc=0.5):
    _noise = L.GaussianNoiseLayer(_incoming, sigma=noise_sigma)
        
    _conv1 = conv(_noise, num_filters=128, filter_size=4, stride=2, pad=1, W=I.Normal(0.02), b=I.Constant(0.), nonlinearity=NL.rectify)
    _lrn1  = L.LocalResponseNormalization2DLayer(_conv1, alpha=0.0001, k=2, beta=0.75, n=5)

    _drop2 = L.DropoutLayer(_lrn1, p=drop_ratio_conv, rescale=True)
    _conv2 = batch_norm_n(conv(_drop2, num_filters=256, filter_size=4, stride=2, pad=1, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))

    _drop3 = L.DropoutLayer(_conv2, p=drop_ratio_conv, rescale=True)
    _conv3 = batch_norm_n(conv(_drop3, num_filters=384, filter_size=3, stride=1, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
      
    _drop4 = L.DropoutLayer(_conv3, p=drop_ratio_conv, rescale=True)
    _conv4 = batch_norm_n(conv(_drop4, num_filters=512, filter_size=3, stride=1, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
      
    _drop5 = L.DropoutLayer(_conv4, p=drop_ratio_fc, rescale=True)
    _conv5 = batch_norm_n(conv(_drop5, num_filters=1024, filter_size=1, stride=1, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
        
    _pool6 = L.GlobalPoolLayer(_conv5, pool_function=theano.tensor.max) #mean
    _drop6 = L.DropoutLayer(_pool6, p=drop_ratio_fc, rescale=True)
    _fc6 = L.DenseLayer(_drop6, ny, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=NL.softmax)
        
    _aux = [_conv1, _conv2, _conv3, _conv4, _conv5, L.DimshuffleLayer(_fc6, (0,1,'x','x'))] ### used to have a tanh() around everything except last
    return _aux, _fc6


###########################################################
##  EMBEDDING DENOISING AUTO-ENCODER
###########################################################
def enc_net(_incoming, output_channels, drop_rate=0.3, nonlinearity=None):
#    #_noise = L.GaussianNoiseLayer(_incoming, sigma=0.1)
    _drop1 = L.DropoutLayer(_incoming, p=drop_rate, rescale=True)
    _fc1 = L.DenseLayer(_drop1, 4*output_channels, W=I.Normal(0.02), b=I.Constant(0.1), nonlinearity=NL.rectify)
    _drop2 = L.DropoutLayer(_fc1, p=drop_rate, rescale=True)
    _fc2 = L.DenseLayer(_drop2, output_channels, W=I.Normal(0.02), b=I.Constant(0.1), nonlinearity=nonlinearity)
    return _fc2

def dec_net(_incoming, output_channels, nonlinearity=None):
    _fc1 = L.DenseLayer(_incoming, 4*_incoming.output_shape[1], W=I.Normal(0.02), b=None, nonlinearity=NL.rectify)
    _fc2 = L.DenseLayer(_fc1, output_channels, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=nonlinearity)
    return _fc2


###########################################################
##  GENERATOR NETWORK
###########################################################
cond_gen_net_str = '4096f_r4_384d3_256d4s2_96d4s2_3d3'

def cond_gen_net(_seed, _cond, num_noise_slices=[0, 0, 0, 0, 0]):
    _seed0, ns = plu.concat_tn(None, _seed, 0, num_noise_slices[0])  
    _seed0 = plu.concat_tc(_seed0, _cond)
    _fc1 = batch_norm(L.DenseLayer(_seed0, 256*4**2, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    _reshape1 = L.ReshapeLayer(_fc1, ([0], 256, 4, 4))
    
    _deconv2, ns = plu.concat_tcn(_reshape1, _cond, _seed, ns, num_noise_slices[1])
    _deconv2 = batch_norm(deconv(_deconv2, num_filters=384, filter_size=3, stride=1, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    
    _deconv3, ns = plu.concat_tcn(_deconv2, _cond, _seed, ns, num_noise_slices[2])
    _deconv3 = batch_norm(deconv(_deconv3, num_filters=256, filter_size=4, stride=2, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    
    _deconv4, ns = plu.concat_tcn(_deconv3, _cond, _seed, ns, num_noise_slices[3])    
    _deconv4  = batch_norm(deconv(_deconv4, num_filters=96, filter_size=4, stride=2, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    
    _deconv5, ns = plu.concat_tcn(_deconv4, _cond, _seed, ns, num_noise_slices[4]) 
    _deconv5  = deconv(_deconv5, num_filters=npc, filter_size=3, stride=1, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.tanh)
    print "===> graph requires nz>=%d <===" % ns
    return _deconv5



###########################################################
##  ENERGY-BASED AUTO-ENCODER
###########################################################
def cond_erg_enc_net(_incoming, _cond, noise_sigma=0.05, drop_ratio_conv=0.1):
    #_noise = L.GaussianNoiseLayer(_incoming, sigma=noise_sigma)
    _drop1 = L.DropoutLayer(_incoming, p=drop_ratio_conv, rescale=True)
    _drop1 = plu.concat_tc(_drop1, _cond)
    _conv1 = batch_norm(conv(_drop1, num_filters=128, filter_size=4, stride=2, pad=1, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.02)))
 
    _drop2 = L.DropoutLayer(_conv1, p=drop_ratio_conv, rescale=True)    
    _drop2 = plu.concat_tc(_drop2, _cond)
    _emb = batch_norm(conv(_drop2, num_filters=256, filter_size=4, stride=2, pad=1, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.02)))
    return _emb

    
def cond_erg_dec_net(_emb, _cond):   
    _deconv2 = plu.concat_tc(_emb, _cond)
    _deconv2 = deconv(_deconv2, num_filters=128, filter_size=4, stride=2, crop=1, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=NL.LeakyRectify(0.02))
    
    _deconv1 = plu.concat_tc(_deconv2, _cond)
    _deconv1 = deconv(_deconv1, num_filters=npc, filter_size=4, stride=2, crop=1, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=None)
    return _deconv1


