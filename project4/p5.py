import cardetect

def train():
    data = cardetect.TrainingSet()
    data.LoadTrainingData()
    detector = cardetect.Detector()
    detector.Train(data.X_train, data.y_train)
    detector.Test(data.X_test, data.y_test)
    detector.Save('model.h5')

def test():
    detector = cardetect.Detector()
    # load picture and run detection
    # TODO: miltiple sizes
    # TODO: sliding windows
    # TODO: heatmap
    # show picture and results

train()
