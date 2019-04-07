#!/usr/bin/env python
# -*- coding: utf-8

# This python script executes the serialconvolution program for all images and kernels in the directories especified by pathImages and pathKernels. The process is done in chunks of partitions variable.
# Results are stored in the pathResult.

# Yo cannot execute this script in the frontend, because it didn't enqueue the executions in the SGE. But, Once you can ensure that your code is working correctly.
# You can modify this script to enqueue different executions to the MOORE's system queue.
import os
#Path for directories
pathImages = '../images/'
pathKernel = '../kernels/'
pathResult = '../results/'
partitions = 1

#Initialize the list of files
lstImages  = []
lstKernels = []
 
#List all files in the directory pathImages
lstDir = os.walk(pathImages)
for root, dirs, files in lstDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
	lstImages.append(nombreFichero+extension)

#List all the file in the directory pathkernel
lstDir = os.walk(pathKernel)
for root, dirs, files in lstDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
	lstKernels.append(nombreFichero+extension)

for img in lstImages:
    for krn in lstKernels:
        #	print(img,krn)
	(namei, extension) = os.path.splitext(img)
	(namek, extension) = os.path.splitext(krn)
	nameimgresults  = namei + namek + '.ppm'
	nametimeresults = namei + namek + '.txt'
    cmd = './serialconvolution ' + pathImages + img + ' ' + pathKernel + krn + ' ' + pathResult + nameimgresults + ' ' + str(partitions) + ' >' + pathResult + nametimeresults
        #    os.system(str)
    print (cmd)

