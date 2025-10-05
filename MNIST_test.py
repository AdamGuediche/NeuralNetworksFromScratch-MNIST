import neuralNetwork as nn
import numpy as np
import json

with open('mnist_handwritten_test.json') as file:
    content = json.loads(file.read())
    X_test=(np.array([number_image['image'] for number_image in content]).astype(np.float32)-127.5)/127.5 # preprocessing pour avoir une des inputs entre -1 et 1
    y_test=np.array([number_image['label'] for number_image in content])

model = nn.Model.load('mnist.model')
model.evaluate(X_test, y_test)