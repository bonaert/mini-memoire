import os
from time import time

import matplotlib.pyplot as plt
import numpy as np


def buildPath(relPath):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dirPath, '..', relPath)


def createDirectoriesIfNeeded(filename):
    dirName = os.path.dirname(filename)
    if not os.path.exists(dirName):
        os.makedirs(dirName)


def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()


# IO

JAFFE_DIR = buildPath('datasets/JAFFE/')
JAFFE_PROCESSED_DIR = buildPath('datasets/transformed/JAFFE/')
COHN_KANADE_DIR = buildPath('datasets/CohnKanadePlus/')
COHN_KANADE_EMOTION_DIR = buildPath('datasets/CohnKanadePlus/cohn-kanade-images')
COHN_KANADE_IMAGES_DIR = buildPath('datasets/CohnKanadePlus/Emotion')

COHN_KANADE_PROCESSED_DIR = buildPath('datasets/transformed/CohnKanadePlus/')
COHN_KANADE_PROCESSED_EMOTION_DIR = buildPath('datasets/transformed/CohnKanadePlus/Emotion')
COHN_KANADE_PROCESSED_IMAGES_DIR = buildPath('datasets/transformed/CohnKanadePlus/cohn-kanade-images')


def getJAFFEImageNames(transformed=True):
    if transformed:
        BASEPATH = JAFFE_PROCESSED_DIR
    else:
        BASEPATH = JAFFE_DIR

    for imageFileName in os.listdir(BASEPATH):
        if 'README' in imageFileName or '.DS' in imageFileName:
            continue

        fullFileName = os.path.join(BASEPATH, imageFileName)
        yield imageFileName, fullFileName


def getCohnKanadeImageNames(transformed=True):
    if transformed:
        BASEPATH = COHN_KANADE_PROCESSED_DIR + 'cohn-kanade-images'
    else:
        BASEPATH = COHN_KANADE_DIR + 'cohn-kanade-images'

    for personDirName in os.listdir(BASEPATH):
        for emotionDirName in os.listdir('%s/%s' % (BASEPATH, personDirName)):
            if 'README' in emotionDirName or '.DS' in emotionDirName:
                continue

            for imageFileName in os.listdir('%s/%s/%s' % (BASEPATH, personDirName, emotionDirName)):
                if 'README' in imageFileName or '.DS' in imageFileName:
                    continue

                fullRelFileName = '%s/%s/%s' % (personDirName,
                                                emotionDirName, imageFileName)
                yield personDirName, emotionDirName, imageFileName, fullRelFileName


def timer(f):
    start = time()
    result = f()
    end = time()
    duration = end - start
    print("It took %.3f seconds" % duration)
    return result


def getAccuracy(predicted, real):
    length = real.shape[0]
    numCorrect = 0
    for i in range(length):
        if np.argmax(predicted[i]) == np.argmax(real[i]):
            numCorrect += 1
    accuracy = numCorrect / length
    return accuracy


def getFreeFileName(nameWithoutExtension, ext='csv'):
    i = 0
    while os.path.exists("%s%s.%s" % (nameWithoutExtension, i, ext)):
        i += 1

    return "%s%s.%s" % (nameWithoutExtension, i, ext)