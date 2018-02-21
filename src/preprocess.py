from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import utils


def transformImage(image):
    greyImage = rgb2gray(image)
    greyImageResized = resize(greyImage, (64, 64))
    return greyImageResized


def transformAndSaveImage(name, datasetDir):
    fileNameWithoutExtension = name.rsplit('.', maxsplit=1)[0]
    image = io.imread(utils.buildPath('datasets/%s/%s' % (datasetDir, name)))
    transformedImage = transformImage(image)

    fullFilePath = utils.buildPath('datasets/transformed/%s/%s.png' %
                                   (datasetDir, fileNameWithoutExtension))
    io.imsave(fullFilePath, transformedImage)


def transformAllImages():
    for imageFileName, fullFileName in utils.getJAFFEImageNames(transformed=False):
        transformAndSaveImage(imageFileName, datasetDir='JAFFE')

    for person, emotion, imageFileName, relFileName in utils.getCohnKanadeImageNames(transformed=False):
        utils.createDirectoriesIfNeeded('%s/%s/%s/test.file' %
                                        (utils.COHN_KANADE_PROCESSED_DIR, person, emotion))
        transformAndSaveImage(relFileName, datasetDir='CohnKanade/cohn-kanade')


if __name__ == '__main__':
    transformAllImages()
