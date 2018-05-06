import os

import numpy as np
from skimage import io
from sklearn.preprocessing import LabelEncoder

import utils


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


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
            path_name = path + '/' + name
            with open(path_name) as f:
                try:
                    line = f.readline()
                except UnicodeDecodeError:
                    print(path_name)
                    raise Exception()

                emotionNum = int(float(line.strip()))
                if oneHotEncoded:
                    return CK_CODED_EMOTIONS[emotionNum - 1]
                else:
                    return emotionNum - 1
    except FileNotFoundError:
        return None

    return None


def getCohnKanadeData(oneHotEncoded=True, useNeutral=False):
    images = []
    emotions = []
    people = []
    cohn_kanade_dir = utils.COHN_KANADE_PROCESSED_IMAGES_DIR

    for personDir in os.listdir(cohn_kanade_dir):
        person_full_dir = cohn_kanade_dir + '/' + personDir
        for emotionDir in os.listdir(person_full_dir):
            emotion_full_dir = person_full_dir + '/' + emotionDir
            imagesNames = sorted(os.listdir(emotion_full_dir))

            emotion = getCKEmotion(personDir, emotionDir, oneHotEncoded)
            if emotion is None:
                continue

            relevantImages = [imagesNames[0], imagesNames[-2], imagesNames[-1]]

            # Last two images to the real emotion
            for i, imageName in enumerate(relevantImages):
                image = getImage(emotion_full_dir + '/' + imageName)
                images.append(image)
                # First image may correspond to neutral
                if useNeutral and i == 0:
                    if oneHotEncoded:
                        emotions.append(CK_NEUTRAL_CODED_EMOTION)
                    else:
                        emotions.append(CK_NEUTRAL_EMOTION_NUM)
                else:
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
JAFFE_CODED_EMOTIONS = [to_categorical(i, JAFFE_NUM_EMOTIONS) for i in range(JAFFE_NUM_EMOTIONS)]
CK_EMOTIONS = [
    'ANGER',
    'CONTEMPT',
    'DISGUST',
    'FEAR',
    'HAPPY',
    'SADNESS',
    'SUPRISE',
]
CK_NUM_EMOTIONS = len(CK_EMOTIONS)
CK_CODED_EMOTIONS = [to_categorical(i, CK_NUM_EMOTIONS) for i in range(CK_NUM_EMOTIONS)]
CK_NEUTRAL_CODED_EMOTION = CK_CODED_EMOTIONS[-1]
CK_NEUTRAL_EMOTION_NUM = CK_NUM_EMOTIONS - 1


def addNeutralEmotion():
    CK_EMOTIONS.append('NEUTRAL')

    global CK_NUM_EMOTIONS, CK_CODED_EMOTIONS, CK_NEUTRAL_CODED_EMOTION, CK_NEUTRAL_EMOTION_NUM
    CK_NUM_EMOTIONS = len(CK_EMOTIONS)
    CK_CODED_EMOTIONS = [to_categorical(i, CK_NUM_EMOTIONS) for i in range(CK_NUM_EMOTIONS)]
    CK_NEUTRAL_CODED_EMOTION = CK_CODED_EMOTIONS[-1]
    CK_NEUTRAL_EMOTION_NUM = CK_NUM_EMOTIONS - 1


def getMode(classifier_predictions_list, oneHotEmotions=None):
    if oneHotEmotions is None:
        if len(classifier_predictions_list[0][0]) == JAFFE_NUM_EMOTIONS:
            oneHotEmotions = JAFFE_CODED_EMOTIONS
        else:
            oneHotEmotions = CK_CODED_EMOTIONS

    counts = sum(classifier_predictions_list)
    maxCountIndex = [np.argmax(row) for row in counts]
    oneHotModes = np.array([oneHotEmotions[i] for i in maxCountIndex])
    return oneHotModes
