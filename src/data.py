import os

import numpy as np
from keras.utils import np_utils
from skimage import io
from sklearn.preprocessing import LabelEncoder

import utils


def getPersonAndEmotionFromFileName(imageName, oneHotEncoded=True):
    person, emotionCode, photoNumber, extension = imageName.split('.')
    for (i, potentialEmotion) in enumerate(JAFFE_EMOTIONS):
        if potentialEmotion in emotionCode:
            if oneHotEncoded:
                return person, JAFFE_CODED_EMOTIONS[i]
            else:
                return person, i

    raise Exception("No emotion matched the file %s" % imageName)


def normalizeImage(image):
    # image = image / 2 ** 16
    # image = (image - image.mean()) / (image.std() + 1e-8)
    return image


def labelEncode(groups):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(groups)


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
    people = []
    cohn_kanade_dir = utils.COHN_KANADE_PROCESSED_IMAGES_DIR

    for personDir in os.listdir(cohn_kanade_dir):
        person_full_dir = cohn_kanade_dir + '/' + personDir
        for emotionDir in os.listdir(person_full_dir):
            emotion_full_dir = person_full_dir + '/' + emotionDir
            imagesNames = sorted(os.listdir(emotion_full_dir))

            relevantImages = [imagesNames[0], imagesNames[-2], imagesNames[-1]]

            emotion = getCKEmotion(personDir, emotionDir, oneHotEncoded)
            if emotion is None:
                continue

            # First image corresponds to neutral
            images.append(getImage(emotion_full_dir + '/' + relevantImages[0]))
            if oneHotEncoded:
                emotions.append(CK_NEUTRAL_CODED_EMOTION)
            else:
                emotions.append(CK_NEUTRAL_EMOTION_NUM)
            people.append(personDir)

            # Last two images to the real emotion
            for imageName in relevantImages[1:]:
                images.append(getImage(emotion_full_dir + '/' + imageName))
                emotions.append(emotion)
                people.append(personDir)

    return np.array(images), np.array(emotions), labelEncode(people)


def getJAFFEData(oneHotEncoded=True):
    images = []
    emotions = []
    people = []
    for imageName, fullPath in utils.getJAFFEImageNames():
        image = getImage(fullPath)
        images.append(image)
        person, emotion = getPersonAndEmotionFromFileName(imageName, oneHotEncoded)
        emotions.append(emotion)
        people.append(person)

    encodedPeople = labelEncode(people)
    return np.array(images), np.array(emotions), encodedPeople


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
    'SUPRISE',
    'NEUTRAL'
]
CK_NUM_EMOTIONS = len(CK_EMOTIONS)
CK_CODED_EMOTIONS = [np_utils.to_categorical(i, CK_NUM_EMOTIONS) for i in range(CK_NUM_EMOTIONS)]
CK_NEUTRAL_CODED_EMOTION = CK_CODED_EMOTIONS[-1]
CK_NEUTRAL_EMOTION_NUM = CK_NUM_EMOTIONS - 1
