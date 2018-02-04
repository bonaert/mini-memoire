from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import os


def train():
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Evaluate your performance in one line:
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

    # Or generate predictions on new data:
    classes = model.predict(x_test, batch_size=128)

def transformImage(image):
    greyImage = rgb2gray(image)
    greyImageResized = resize(greyImage, (64, 64))
    return greyImageResized

def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def buildPath(relPath):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dirPath, '..', relPath)

def transformAndSaveImage(name, dataset='JAFFE'):
    fileNameWithoutExtension = name.rsplit('.', maxsplit=1)[0]
    image = io.imread(buildPath('datasets/%s/%s' % (dataset, name)))
    transformedImage = transformImage(image)
    io.imsave(buildPath('datasets/transformed/%s/%s.png' % (dataset, fileNameWithoutExtension)), transformedImage)


def loadImages():
    print("Starting to read image")
    image = io.imread(buildPath('datasets/JAFFE/KA.AN1.39.tiff'))
    print("Stopped reading iamge")
    transformedImage = transformImage(image)
    
    print(transformedImage[10, 10])
    #showImage(transformedImage)
    

    io.imsave(buildPath('datasets/transformed/test.png'), transformedImage)


    # image.resize()
    print(image.shape)


def transformAllImages():
    for imageFileName in os.listdir(buildPath('datasets/JAFFE/')):
        if 'README' in imageFileName or '.DS' in imageFileName:
            continue

        transformAndSaveImage(imageFileName)

#loadImages()
transformAllImages()
