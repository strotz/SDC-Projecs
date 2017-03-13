import cardetect

import os
os.system('set CUDA_VISIBLE_DEVICES=""')

def train():
    data = cardetect.TrainingSet()
    data.LoadTrainingData()
    detector = cardetect.Detector()
    detector.Train(data.X_train, data.y_train)
    detector.Test(data.X_test, data.y_test)
    detector.Save('model.h5')

train()
