import shutil

from tqdm import tqdm
from PIL import Image

import utils


def transformImage(image):
    # greyImage = rgb2gray(image)
    # image_resized = resize(image, (64, 64), anti_aliasing=True)
    image = image.convert('L')
    return image.resize((64, 64), resample=Image.LANCZOS)
    # return image_resized


def transformAndSaveImage(name, datasetDir):
    fileNameWithoutExtension = name.rsplit('.', maxsplit=1)[0]
    filePath = utils.buildPath('datasets/%s/%s' % (datasetDir, name))

    image = Image.open(filePath)
    transformedImage = transformImage(image)

    fullFilePath = utils.buildPath('datasets/transformed/%s/%s.png' %
                                   (datasetDir, fileNameWithoutExtension))
    utils.createDirectoriesIfNeeded(fullFilePath)
    transformedImage.save(fullFilePath, format="png")


def transformAllImages():
    for imageFileName, fullFileName in tqdm(utils.getJAFFEImageNames(transformed=False)):
        transformAndSaveImage(imageFileName, datasetDir='JAFFE')

    for person, emotion, imageFileName, relFileName in tqdm(utils.getCohnKanadeImageNames(transformed=False)):
        utils.createDirectoriesIfNeeded('%s/%s/%s/test.file' %
                                        (utils.COHN_KANADE_PROCESSED_IMAGES_DIR, person, emotion))
        transformAndSaveImage(relFileName, datasetDir='CohnKanadePlus/cohn-kanade-images')


    # Copy emotions
    shutil.copytree(src=utils.COHN_KANADE_EMOTION_DIR, dst=utils.COHN_KANADE_PROCESSED_EMOTION_DIR)


if __name__ == '__main__':
    transformAllImages()
