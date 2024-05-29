## JDONNN - Joint Detection Of Neural-Network-Noise
An intentionally basic script that aims to detect if an image was generated with image-diffusion. I specifically constrained the detection to Structural Similarity Index and image Entropy (hence, 'Joint'), hoping to 'extract' a noisy pattern that could be detected over a regular non-diffused image. On some batches of images the success rate nears 90%, on others it lags at just 60-70%. I just wanted to throw something on Github so I can tell myself I didn't waste my time.

### Requirements

* opencv-python
* scikit-image
* numpy

## Usage

This script can be placed anywhere. By default with no parameters, it'll run on every image in the same directory as it, excluding subdirectories. 

```
usage: detectimages.py [-h] [-filepath FILEPATH] [-image IMAGE] [-test] [-concurrent] [-best]

options:
  -h, --help            show this help message and exit
  -filepath FILEPATH, -f FILEPATH
                        Filepath to test. Without any args, this script tests the folder in which the .py is placed.
  -image IMAGE, -i IMAGE
                        Specifies a lone image to calculate. You can add a filepath. Eg: -f A:/path/ -i image001.png |
                        If your image contains spaces, place it in double quotes.
  -test, -t             Gives detailed data on images for testing accuracy. Requires renaming all ai images to contain
                        'aiimg'
  -concurrent, -c       Uses concurrency to hopefully speed up performance. Only useful for multiple images.
  -best, -b             Use with -test. Will run through the image(s) with different parameters until it achieves the
                        best result
```

## Results

There might be something in using SSIM and entropy to detect image diffusion, or there might not be. Most other open-source detection projects used pre-trained models, in a 'fight fire with fire' sort of way. I'm too dumb to do that so I stuck with basically trying to see if the 'noise' that results from image-diffusion could be amplified and detected compared to regular images. Many 'user friendly' online tools were proprietary so who knows what they use? There are a few 'seeds' of values (hence the usage of -best) which vary the output of a batch of images. Further, the type of image affects the output, such as a blurry photo, or high quality drawing, or an AI image that has been compressed into .webp multiple times. There might be something with these values, I have no idea nor should something like this realistically be investigated.