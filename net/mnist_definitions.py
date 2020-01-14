#########################################################################
#       Set of networks designed for MNIST
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


npc = 1         # # of channels in image
npx = 28        # # of pixels width/height of images
ny = 10         # # of classes

cls_net_str = 'mnist_cls_net'
# 64c7s3_128c3_mp2_256f_10f

def cls_net(_incoming):
    _drop1 = L.DropoutLayer(_incoming, p=0.2, rescale=True)
    _conv1 = batch_norm(conv(_drop1, num_filters=64, filter_size=7, stride=3, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))  
    _drop2 = L.DropoutLayer(_conv1, p=0.2, rescale=True)
    _conv2 = batch_norm(conv(_drop2, num_filters=128, filter_size=3, stride=1, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    _pool2 = L.MaxPool2DLayer(_conv2, pool_size=2)     
    _fc1 = batch_norm(L.DenseLayer(L.FlattenLayer(_pool2, outdim=2), 256, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))

    _fc2 = L.DenseLayer(_fc1, ny, W=I.Normal(0.02), b=None, nonlinearity=NL.sigmoid) 
    _aux = [tanh(_conv1), tanh(_conv2), tanh(L.DimshuffleLayer(_fc1, (0,1,'x','x'))), L.DimshuffleLayer(_fc2, (0,1,'x','x')) ] 
    return _aux, _fc2    


def enc_net(_incoming, output_channels, drop_rate=0.3, nonlinearity=None):
    #_noise = L.GaussianNoiseLayer(_incoming, sigma=0.1)
    _drop1 = L.DropoutLayer(_incoming, p=drop_rate, rescale=True)
    _fc1 = batch_norm(L.DenseLayer(_drop1, 2*output_channels, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=NL.rectify))
    _drop2 = L.DropoutLayer(_fc1, p=drop_rate, rescale=True)
    _fc2 = L.DenseLayer(_drop2, output_channels, W=I.Normal(0.02), b=I.Constant(0.1), nonlinearity=nonlinearity)
    return _fc2

def dec_net(_incoming, output_channels, nonlinearity=None):
    _fc1 = batch_norm(L.DenseLayer(_incoming, 2*_incoming.output_shape[1], W=I.Normal(0.02), b=I.Constant(0), nonlinearity=NL.rectify))
    _fc2 = L.DenseLayer(_fc1, output_channels, W=I.Normal(0.02), b=I.Constant(0), nonlinearity=nonlinearity)
    return _fc2



gen_net_str = 'mnist_gen_net'

def gen_net(_seed, num_noise_slices, _cond=None):
    _fc1, ns = plu.concat_tn(None, _seed, 0, num_noise_slices[0])
    if _cond!=None:
        _fc1 = plu.concat_tc(_fc1, _cond)
    _fc1 = batch_norm(L.DenseLayer(_fc1, 1024, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    #
    if _cond!=None:
        _fc2, ns = plu.concat_tcn(_fc1, _cond, _seed, ns, num_noise_slices[1])
    else:
        _fc2, ns = plu.concat_tn(_fc1, _seed, ns, num_noise_slices[1])
    #
    _fc2 = batch_norm(L.DenseLayer(_fc2, 128*4**2, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    _reshape2 = L.ReshapeLayer(_fc2, ([0], 128, 4, 4))
    if _cond!=None:
        _reshape2, ns = plu.concat_tcn(_reshape2, _cond, _seed, ns, num_noise_slices[2])
    else:
        _reshape2, ns = plu.concat_tn(_reshape2, _seed, ns, num_noise_slices[2])
    #
    _deconv3 = batch_norm(deconv(_reshape2, num_filters=128, filter_size=3, stride=1, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    if _cond!=None:
        _deconv4, ns = plu.concat_tcn(_deconv3, _cond, _seed, ns, num_noise_slices[3])
    else:
        _deconv4, ns = plu.concat_tn(_deconv3, _seed, ns, num_noise_slices[3])
    #
    _deconv4 = batch_norm(deconv(_deconv4, num_filters=64, filter_size=4, stride=2, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.rectify))
    if _cond!=None:
        _deconv5, ns = plu.concat_tcn(_deconv4, _cond, _seed, ns, num_noise_slices[4])
    else:
        _deconv5, ns = plu.concat_tn(_deconv4, _seed, ns, num_noise_slices[4])
    _deconv5 = deconv(_deconv5, num_filters=npc, filter_size=4, stride=2, crop=1, W=I.Normal(0.02), b=None, nonlinearity=NL.sigmoid)
    print "===> graph requires nz>=%d <===" % ns
    return _deconv5    
    

def erg_enc_net(_input, _cond=None):
    if _cond!=None:
        _input = plu.concat_tc(_input, _cond)
    _conv1 = batch_norm(conv(_input, num_filters=64, filter_size=4, stride=2, pad=1, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.1)))
    #
    if _cond!=None: 
        _conv1 = plu.concat_tc(_conv1, _cond)
    _conv2 = batch_norm(conv(_conv1, num_filters=128, filter_size=4, stride=2, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.1)))
    #
    if _cond!=None:
        _conv2 = plu.concat_tc(_conv2, _cond)
    _emb = batch_norm(conv(_conv2, num_filters=256, filter_size=3, stride=1, pad=0, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.1)))
    return _emb

def erg_dec_net(_emb, _cond):
    if _cond!=None: 
        _conv3 = plu.concat_tc(_emb, _cond)    
    else:
        _conv3 = _emb
    _deconv1 = batch_norm(deconv(_conv3, num_filters=128, filter_size=3, stride=1, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.01)))
    #      
    if _cond!=None: 
        _deconv1 = plu.concat_tc(_deconv1, _cond)
    _deconv2 = deconv(_deconv1, num_filters=64, filter_size=4, stride=2, crop=0, W=I.Normal(0.02), b=None, nonlinearity=NL.LeakyRectify(0.01))
    #
    if _cond!=None:
        _deconv2 = plu.concat_tc(_deconv2, _cond)
    _deconv3 = deconv(_deconv2, num_filters=npc, filter_size=4, stride=2, crop=1, W=I.Normal(0.02), b=None, nonlinearity=None)
    return _deconv3

