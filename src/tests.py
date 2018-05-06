import numpy as np
import unittest

from data import getMode


class Tests(unittest.TestCase):
    def testGetModeSimple(self):
        oneHotEmotions = np.array([[1, 0], [0, 1]])

        classifierPred = [[1, 0]]
        predictions = np.array([classifierPred])

        mode = getMode(predictions, oneHotEmotions)
        self.assertTrue(np.array_equal(mode, classifierPred))

    def testGetModeComplex(self):
        oneHotEmotions = np.array([[1, 0], [0, 1]])

        pred1 = [[1, 0], [1, 0], [0, 1]]
        pred2 = [[1, 0], [0, 1], [0, 1]]
        pred3 = [[1, 0], [0, 1], [1, 0]]
        predictions = np.array([pred1, pred2, pred3])

        expected = [[1, 0], [0, 1], [0, 1]]

        mode = getMode(predictions, oneHotEmotions)
        self.assertTrue(np.array_equal(mode, expected))
