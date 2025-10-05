import neuralNetwork as nn
import numpy as np
import json

with open('mnist_handwritten_train.json') as file:
    content = json.loads(file.read())
    X_tot=(np.array([number_image['image'] for number_image in content]).astype(np.float32)-127.5)/127.5 # preprocessing pour avoir une des inputs entre -1 et 1
    y_tot=np.array([number_image['label'] for number_image in content])

X=X_tot[:50000]
X_val = X_tot[50000:]
y=y_tot[:50000]
y_val=y_tot[50000:]

mod = nn.Model()
mod.add(nn.Layer_Dense(784, 512))
mod.add(nn.Layer_Dropout(0.1))
mod.add(nn.Activation_ReLU())
mod.add(nn.Layer_Dense(512, 512))
mod.add(nn.Layer_Dropout(0.1))
mod.add(nn.Activation_ReLU())
mod.add(nn.Layer_Dense(512, 10))
mod.add(nn.Activation_Softmax())

mod.set(loss=nn.Loss_CategoricalCrossentropy(), optimizer=nn.Optimizer_Adam(decay=1e-3), accuracy=nn.Accuracy_Categorical(False))
mod.finalize()
mod.train(X,y,epochs = 10, batch_size=128, print_every=100, validation_data=(X_val, y_val))
mod.save('mnist.model')

