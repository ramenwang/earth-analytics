# RS Super-Resolution
* The model upsamples the images from rougher spatial resolution to finer. 

## Intuition
The model consists of three main parts: feature extractor, feature selector, image assembler. The model input is the low-resolutionn image, and target is the residual between true high-resolution image and bicubic upsampled image. In the forward pass, <br>
1. the input image is first digested by feature extractor that creates a stack of feature maps at different spatial scales <br>
2. the feature selector working on feature stack to select responsible feature maps <br>
3. the selected feature maps are reassembled to form the target image (residual image)
<img src="https://github.com/ramenwang/earth-analytics/blob/master/RSSupperRes/pics/cnn_arch.png" width="600" align='center'>

## Results
<img src="https://github.com/ramenwang/earth-analytics/blob/master/RSSupperRes/pics/result_1.png" width="600" align='center'>
<img src="https://github.com/ramenwang/earth-analytics/blob/master/RSSupperRes/pics/result_2.png" width="600" align='center'>

## Current available models
1. 2x resolution upsampling consists of:
> a 9-layer feature extractor with number of filters [128-109-82-56-38-28-24-24-24]
> a 3-path inception feature selector with number of filters [32-16-16]
> a 1-layer image reconstructor with number of filters [24]
