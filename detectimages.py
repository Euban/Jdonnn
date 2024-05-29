import os
import concurrent
from concurrent import futures
import math

import cv2
from skimage.metrics import structural_similarity as ssim
import skimage
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-filepath', '-f', type=str, help="Filepath to test. Without any args, this script tests the folder in which the .py is placed.")
parser.add_argument('-image', '-i', type=str, help='Specifies a lone image to calculate. You can add a filepath. Eg: -f A:/path/ -i image001.png | If your image contains spaces, place it in double quotes.')
parser.add_argument('-test', '-t', help="Gives detailed data on images for testing accuracy. Requires renaming all ai images to contain 'aiimg'", action='store_true')
parser.add_argument('-concurrent', '-c', help="Uses concurrency to hopefully speed up performance. Only useful for multiple images.", action='store_true')
parser.add_argument('-best', '-b', help="Use with -test. Will run through the image(s) with different parameters until it achieves the best result", action='store_true')
args = parser.parse_args()

#By default set to false since we just run through once, only set true if -test and -best flags are enabled
runUntilBest = False

if args.test:
    knowData = True
    #We can only test for the best list of vars if we are testing on known images.
    if args.best:
        runUntilBest = True   
else:
    knowData = False

#If user selects filepath and that file exists, that becomes the dir, if the path is invalid, raise error, if no path is specified, the dir of the .py file is the working dir
if args.filepath:
    if os.path.isdir(args.filepath):
        cwd = args.filepath
    elif os.path.isdir(args.filepath) == False:
        raise argparse.ArgumentTypeError('Directory not found.')
else:
    cwd = os.getcwd()

#Just so the filepath works with each image. A:/pathimage.png would happen without, instead of A:/path/image.png
if str(cwd).endswith('/') == False:
    cwd += '/'

#If user specifies a sole image in this or any dir, set it here.
if args.image: 
    if os.path.exists(os.path.join(cwd, args.image)):
        imgfile = args.image
    elif os.path.exists(os.path.join(cwd, args.image) == False):
        raise argparse.ArgumentTypeError('Image not found in directory.')
else:
    imgfile = None

#To use concurrency, which I have no idea how it works lmao
if args.concurrent:
    useConcurrent = True
else:
    useConcurrent = False

#Lists of images processed, their SSIM and entropy values.
global SSIMList, entropyList, imagelist
SSIMList = []
entropyList = []
imagelist = []

#Here are a few lists of values I generated using a random function, often on widely different datasets usually consisting of 50% ai and 50% real, sometimes photos, sometimes art, etc. Which is best? None.
#0 is probably the best performing. Better results, albeit worse in some ways. But it's more likely when it makes a mistake to classify a real image as AI than an AI image as real
#0 - 3.1036741671567816,-2.9432536108558764,1200,45,0.08890861318806809,0.12632463498817526,38,5.358775796960636,1.6565857005777291,10
#1 - 3.1109107558289284,0.07809043706361353,1230,54,0.09816896165701598,0.09181123141719955,41,5.463688517638827,1.5974808994303011,8
#2 - 3.1029987814107396,1.0334491486547712,1200,53,0.07020038723691907,0.11889147101099007,44,5.310442378080723,1.6247623189499483,9
#3 - 3.1036741671567816,-2.9432536108558764,1200,45,0.08890861318806809,0.12632463498817526,34,5.659647425473963,1.6018845960383186,9
#4 - 3.153443669528754,0.001199955196327913,1165,51,0.09577661120786417,0.10157256226372016,45,5.075345621286079,1.6387548942058863,7
#5 - 3.034106307340338,0.06317176713634083,1235,39,0.11143427830451708,0.07721735871823054,34,5.245806639216278,1.5942600905404072,9

#List of variables that determine how well it performs. Seriously, these numbers are magic. On some test cases I get upwards ot 90%. Sometimes it guesses all AI images as AI, all art/drawings as Real, and all photos as AI. Or the opposite or any other combination. The numbers Mason, what do they mean? 
#Using -best, we can run through this select list of variables to find the best one.
def initvars(j):
    global Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby,sigma
    match j:
        case 0:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.1036741671567816,-2.9432536108558764,1200,45,0.08890861318806809,0.12632463498817526,38,5.358775796960636,1.6565857005777291,10
        case 1:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.1109107558289284,0.07809043706361353,1230,54,0.09816896165701598,0.09181123141719955,41,5.463688517638827,1.5974808994303011,8
        case 2:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.1029987814107396,1.0334491486547712,1200,53,0.07020038723691907,0.11889147101099007,44,5.310442378080723,1.6247623189499483,9
        case 3:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.1036741671567816,-2.9432536108558764,1200,45,0.08890861318806809,0.12632463498817526,34,5.659647425473963,1.6018845960383186,9
        case 4:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.153443669528754,0.001199955196327913,1165,51,0.09577661120786417,0.10157256226372016,45,5.075345621286079,1.6387548942058863,7
        case 5:
            Confidencedivby,Confidencelimit,SSIMconfmultby,EntropyLimmultby,SSIMLim,EntropyLim,mean,var,basesigma,Imgsizeweightdivby = 3.034106307340338,0.06317176713634083,1235,39,0.11143427830451708,0.07721735871823054,34,5.245806639216278,1.5942600905404072,9
    sigma = var**basesigma

def makeGuassianNoise(image):
    #https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    #Here's the application of guassian noise. Mean, var, and sigma influence the noised image, and could be tweaked more to change the noise. Returns the modified noised image
    row,col,ch = image.shape
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    imageB = image + gauss
    imageB = imageB.astype(np.uint8)
    return imageB

#Converts image to grayscale and then runs Structural Similarity between the base image and its noised counterpart. 
def calcSSIMScore(image, imageB):
    #Convert to grayscale for SSIM
    grayA = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    #Structural Similarity Index (SSIM) between the 2 images
    (score, diff) = ssim(grayA, grayB, full=True)
    return score

def reset():
    global SSIMList, entropyList, imagelist
    SSIMList = []
    entropyList = []
    imagelist = []

def calcDiffs(file, cwd): 
    suffix = ('.png'), ('.jpg'), ('.jpeg'), ('.webp'), ('.tiff'), ('.tif'), ('.bmp'), ('jfif'), ('avif'), ('.PNG'), ('.JPG'), ('.WEBP'), ('.TIFF'), ('.TIF'), ('.BMP'), ('JFIF'), ('AVIF')
    #If file is an image, add it to the list of images processed and then load it into CV2, make a noised version, and calculate the difference in SSIM values between those two images.
    if file.endswith(suffix):  
        imagelist.append(str(file)) 
        image = cv2.imdecode(np.fromfile(cwd + file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        imageB = makeGuassianNoise(image)
        score = calcSSIMScore(image, imageB)
        #To counteract the tendency for large images to be flagged as AI, we try and reduce large images based on their pixel size.
        height, width = image.shape[:2]
        pixels = height * width
        #No idea what logs are really but I think this works
        Imgsizeweight = (math.log10(pixels)/Imgsizeweightdivby)
        #Add the score to the list of SSIM scores
        SSIMList.append(score * (1.0-Imgsizeweight))

        #Down here we do the same except we don't need grayscale images. Calc the difference between entropy values of the two images.
        #-Entropy-#
        #NOTE: Changed entropy measure to original image and imageB rather than grayscale, but further testing could be done.
        entropy1 = skimage.measure.shannon_entropy(image)
        entropy2 = skimage.measure.shannon_entropy(imageB)
        #Currently, the entropy 'score' is merely the difference between the two images, although more could be done to scale it in the future. 
        entropyDiff = entropy2 - entropy1
        entropyList.append(entropyDiff)
        print('processed ' + str(file))

def main():
    #Keeping track of the best parameters
    mostcorrect = 0
    bestseed = 0
    #Main loop for -best flag, exits early if flag isn't set
    for j in range(5):
        imgprocessed = 0
        imgguessai = 0
        imgguessreal = 0
        #We need to reset the list of images, ssim, and entropy values before redoing the calculations.
        reset()
        #The following vars here are just for benchmarking.
        guessaibutisreal = 0
        guessrealbutisai = 0
        correct = 0
        wrong = 0
        SSIMai = 0
        SSIMreal = 0
        aiimgs = 0
        realimgs = 0
        entropyAI = 0
        entropyREAL = 0
 
        #We initialize variables, sending a 0 for the first list of vars. If -best flag is set, we iterate through different lists of values. 
        initvars(j)   
        if imgfile == None:
            #For each file with a image extension, assuming that the user didn't input a sole image, calculate SSIM and Entropy between the image and a guassian noised version. Does this actually help identify AI images or is this just snake oil? Only one way to find out!
            #If concurrent is enabled with -c, we create 10 workers to iterate through the file list
            if useConcurrent:
                executor = concurrent.futures.ThreadPoolExecutor(10)
                futures = [executor.submit(calcDiffs, file, cwd) for file in os.listdir(cwd)]
                concurrent.futures.wait(futures)
            #Without -c, we just simply iterate through the file list one at a time
            else:
                for file in os.listdir(cwd):
                    calcDiffs(file, cwd)
        else:
            #If one image is selected, we process it here, without concurrency because it's just one image.
            calcDiffs(imgfile, cwd)

        #Calculating the binary ai or real results based on the SSIM and Entropy. Specifically, images that are further from the SSIMLim and EntropyLim have higher confidence, wheras images with SSIMLIM and EntropyLim values close to the limit have low confidence.
        for i in range (len(imagelist)):
            #Gets the confidence by examining how far away from the cutoff point the SSIM value is.
            SSIMConf = round((SSIMList[i] - (SSIMLim)) * SSIMconfmultby, 3)
            #Since Entropy is backwards (lower value is more AI like, and SSIM is higher value is more AI like, we have to subtract the two and minus)
            EntropyConf = round((entropyList[i] - (EntropyLim)) * EntropyLimmultby, 3)

            #If an average of the two confidences (SSIM and entropy) is greater than the confidence limit, it's determined as image-diffusion, else it's real. Simple? Yes. Effective? No.
            if ((SSIMConf - EntropyConf) / Confidencedivby > Confidencelimit): 
                print(str(imagelist[i]) + "'s SSIM is: " + str(SSIMList[i]) + ' and entropy is: ' + str(entropyList[i]) + ' thus is AI.')
                imgguessai += 1
                #Sorting images, again, only needed if benchmarking results
                if ('aiimg' in str(imagelist[i])):
                    correct += 1
                else:
                    guessaibutisreal += 1
                    wrong += 1
            else:
                print(str(imagelist[i]) + "'s SSIM is: " + str(SSIMList[i]) + ' and entropy is: ' + str(entropyList[i]) + ' thus is REAL.')
                imgguessreal += 1
                #Sorting for benchmarking
                if ('aiimg' in str (imagelist[i])):
                    wrong += 1
                    guessrealbutisai += 1
                else:
                    correct += 1
            #Adding logging data to images if they match the critera.
            if ('aiimg' in str(imagelist[i])):
                SSIMai += SSIMList[i]
                aiimgs += 1
                entropyAI += entropyList[i]
            else:
                SSIMreal += SSIMList[i]
                realimgs += 1
                entropyREAL += entropyList[i]
            imgprocessed += 1

        #Only send out data about the accuracy of the script if the flag -test is set.
        #For multiple images
        if knowData and imgfile == None:
            print('Total images: ' + str(imgprocessed) + '. Guessed ' + str(correct) + ' correctly, and got ' + str(wrong) + ' wrong. ' + str((correct/imgprocessed) * 100) + '% correct')     
            print('Further: Thought ' + str(guessaibutisreal) + ' REAL images were AI. And that ' + str(guessrealbutisai) + ' AI images were REAL')  
            print('Average AI SSIM: ' + str(SSIMai / aiimgs))
            print('Average REAL SSIM: ' + str(SSIMreal / realimgs))
            print('Difference in SSIM between AI and REAL: ' + str((SSIMai / aiimgs) - (SSIMreal / realimgs)))
            print('Average AI Entropy: ' + str(entropyAI / aiimgs))
            print('Average REAL Entropy: ' + str(entropyREAL / realimgs))
        #For just one image with -test
        elif knowData:
            print('Guessed ' + str(correct) + ' correctly, and got ' + str(wrong) + ' wrong. ')    
            print('AI SSIM: ' + str(SSIMai)) 
            print('REAL SSIM: ' + str(SSIMreal))
            print('AI Entropy: ' + str(entropyAI))
            print('REAL Entropy: ' + str(entropyREAL)) 
        #No -test flag
        else:
            print ('Processed ' + str(imgprocessed) + ' images. Guessed ' + str(imgguessai) + ' images as AI, and ' + str(imgguessreal) + ' images as real.')

        #For -best, if this generation produced the best results, set it here.
        if mostcorrect < correct:
            mostcorrect = correct
            bestseed = j

        if runUntilBest == False:
            break
        
    if runUntilBest == True:
        print("The highest success rate was " + str(mostcorrect) + " correct guesses with the seed at index " + str(bestseed))

if __name__ == "__main__":
    main()