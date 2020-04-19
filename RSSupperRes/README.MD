# RS Super-Resolution
* The model upsamples the images from rougher spatial resolution to finer. 
## Intuition
The model consists of 
## Current available models
1. 2x resolution upsampling consists of:
> a 9-layer feature extractor with number of filters [48, 47, 46, 44, 40, 34, 25, 16, 12]
> a 3-path inception feature selector with filter size [(1,1), (3,3), (5,5)]
> a 1-layer image reconstructor with number of filters 24
