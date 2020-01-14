# Generative Adversarial Networks Conditioned on Brain Activity Reconstruct Seen Images
Preprint: https://www.biorxiv.org/content/early/2018/04/20/304774
Reprint: https://ieeexplore.ieee.org/document/8616183

## Supplementary Material
The thumbnail images below show the pixel-wise average over all frames of the associated video.

### File: [surrogate_ext_samples_Apr-14-2018_1615.mp4](/surrogate_ext_samples_Apr-14-2018_1615.mp4)
![surrogate_ext_samples_Apr-14-2018_1615.mp4](/fig/surrogate_ext_sample_average_Apr-14-2018_1615.png)
This shows a composite video of the trained generator samples conditioned on the surrogate "ext" images (See Fig. 1 for a description of the datasets). This demonstrate the limit of the reconstruction accuracy under our method.

### File: [vim-1_val_samples_Apr-14-2018_1615.mp4](/vim-1_val_samples_Apr-14-2018_1615.mp4)
![vim-1_val_samples_Apr-14-2018_1615.mp4](/fig/vim-1_val_sample_average_Apr-14-2018_1615.png)
This shows a composite video of the trained generator samples conditioned on the denoised vim-1 "val" voxel set corresponding to Fig. 4. This demonstrate the generality of the decoder and the limit of reconstruction accuracy given successful denoising. 

### File: [vim-1_test_samples_Apr-14-2018_1615.mp4](/vim-1_test_samples_Apr-14-2018_1615.mp4)
![vim-1_test_samples_Apr-14-2018_1615.mp4](/fig/vim-1_test_sample_average_Apr-14-2018_1615.png)
This shows a composite video of the trained generator samples conditioned on the fully cross validated vim-1 "test" voxel set corresponding to Fig. 5.

## Details of the implementation
### Encoding
[Learning the feature extractor](/gan_imaging_cifar-10_classifier.ipynb)

[Learning the voxel encoding model](/gan_imaging_cifar-10_fwrf_training.ipynb)

### Decoding
[Learning the conditional generative model and sampling](/gan_imaging_cifar-10.ipynb)
