import os

import matplotlib.pyplot as plt


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
COHN_KANADE_DIR = buildPath('datasets/CohnKanade/cohn-kanade')
COHN_KANADE_PROCESSED_DIR = buildPath(
    'datasets/transformed/CohnKanade/cohn-kanade')


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
        BASEPATH = COHN_KANADE_DIR
    else:
        BASEPATH = COHN_KANADE_PROCESSED_DIR

    for personDirName in os.listdir(BASEPATH):
        for emotionDirName in os.listdir('%s/%s' % (BASEPATH, personDirName)):
            for imageFileName in os.listdir('%s/%s/%s' % (BASEPATH, personDirName, emotionDirName)):
                fullRelFileName = '%s/%s/%s' % (personDirName,
                                                emotionDirName, imageFileName)
                yield personDirName, emotionDirName, imageFileName, fullRelFileName
