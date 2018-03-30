import os

import numpy as np
from keras.utils import np_utils
from skimage import io

import utils


def getEmotionFromFileName(imageName, oneHotEncoded=True):
    person, emotionCode, photoNumber, extension = imageName.split('.')
    for (i, potentialEmotion) in enumerate(JAFFE_EMOTIONS):
        if potentialEmotion in emotionCode:
            if oneHotEncoded:
                return JAFFE_CODED_EMOTIONS[i]
            else:
                return i

    raise Exception("No emotion matched the file %s" % imageName)


def normalizeImage(image):
    image = image / 2 ** 16
    return image
    # return (image - image.mean()) / (image.std() + 1e-8)


def getCKEmotion(personDir, emotionDir, oneHotEncoded):
    cohn_kanade_emotions_dir = utils.COHN_KANADE_PROCESSED_EMOTION_DIR
    path = cohn_kanade_emotions_dir + '/' + personDir + '/' + emotionDir

    try:
        for name in os.listdir(path):
            with open(path + '/' + name) as f:
                line = f.readline()
                emotionNum = int(float(line.strip()))
                if oneHotEncoded:
                    return CK_CODED_EMOTIONS[emotionNum - 1]
                else:
                    return emotionNum - 1
    except FileNotFoundError:
        return None

    return None


def getCohnKanadeData(oneHotEncoded=True):
    images = []
    emotions = []
    cohn_kanade_dir = utils.COHN_KANADE_PROCESSED_IMAGES_DIR

    for personDir in os.listdir(cohn_kanade_dir):
        person_full_dir = cohn_kanade_dir + '/' + personDir
        for emotionDir in os.listdir(person_full_dir):
            emotion_full_dir = person_full_dir + '/' + emotionDir
            imagesNames = sorted(os.listdir(emotion_full_dir))

            relevantImages = {imagesNames[0], imagesNames[-2], imagesNames[-1]}

            emotion = getCKEmotion(personDir, emotionDir, oneHotEncoded)
            if emotion is None:
                continue

            for imageName in relevantImages:
                images.append(getImage(emotion_full_dir + '/' + imageName))
                emotions.append(emotion)

    return np.array(images), np.array(emotions)


def getJAFFEData(oneHotEncoded=True):
    images = []
    emotions = []
    for imageName, fullPath in utils.getJAFFEImageNames():
        image = getImage(fullPath)
        images.append(image)
        emotions.append(getEmotionFromFileName(imageName, oneHotEncoded))

    return np.array(images), np.array(emotions)
    # NUMDATA = 800
    # return np.array(images[:NUMDATA]), np.array(emotions[:NUMDATA])


def getImage(fullPath):
    return normalizeImage(io.imread(fullPath))


JAFFE_EMOTIONS = [
    'AN',  # ANGER
    'DI',  # DISGUST,
    'FE',  # FEAR
    'HA',  # HAPPINESS
    'NE',  # NEUTRAL
    'SA',  # SADNESS
    'SU'  # SURPRISE
]
JAFFE_NUM_EMOTIONS = len(JAFFE_EMOTIONS)
JAFFE_CODED_EMOTIONS = [np_utils.to_categorical(i, JAFFE_NUM_EMOTIONS) for i in range(JAFFE_NUM_EMOTIONS)]
CK_EMOTIONS = [
    'ANGER',
    'CONTEMPT',
    'DISGUST',
    'FEAR',
    'HAPPY',
    'SADNESS',
    'SUPRISE'
]
CK_NUM_EMOTIONS = len(CK_EMOTIONS)
CK_CODED_EMOTIONS = [np_utils.to_categorical(i, CK_NUM_EMOTIONS) for i in range(CK_NUM_EMOTIONS)]


# TODO: CK and JAFFE have the same amount of labels, so I use the JAFFE coded emotions everywhere
# even though it's a bit hackish