import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import nnfs
from nnfs.datasets import spiral_data, sine_data
import os
import cv2
import json
import pickle
import copy

nnfs.init()

class Activation_ReLU():

    def forward(self, batch, training=False):
        self.inputs = batch
        self.output = np.maximum(batch, 0)

    def backward(self, grad):
        self.dinputs = (self.inputs > 0)*grad


class Layer_Dense():

    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1, self.weight_regularizer_l2, self.bias_regularizer_l1, self.bias_regularizer_l2 = weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2

    def forward(self, batch, training=False):
        self.inputs = batch
        self.output = np.dot(batch, self.weights)+self.biases

    def backward(self, grad):
        self.dinputs = np.dot(grad, self.weights.T)

        self.dweights = np.dot(self.inputs.T, grad) + \
            self.weight_regularizer_l1
        self.dbiases = np.sum(grad, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dweight_l1 = np.ones_like(self.weights)
            dweight_l1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dweight_l1

        if self.weight_regularizer_l2 > 0:
            self.dweights += self.weight_regularizer_l2 * 2*self.weights

        if self.weight_regularizer_l1 > 0:
            dbias_l1 = np.ones_like(self.biases)
            dbias_l1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dbias_l1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += self.bias_regularizer_l2 * 2*self.biases
            
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases=biases


class Activation_Sigmoid():

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues*self.output*(1-self.output)

    def predictions(self, outputs):
        return (outputs > 0.5)*1


class Activation_Softmax():

    def forward(self, batch, training=False):
        self.batch = batch
        exp_batch = np.exp(batch-np.max(batch, axis=1, keepdims=True))
        self.output = exp_batch/np.sum(exp_batch, axis=1, keepdims=True)

    def backward(self, grad):
        grad_inputs = np.empty_like(grad)

        for i, (grad_i, output) in enumerate(zip(grad, self.output)):
            output = output.reshape(-1, 1)
            grad_inputs[i] = (np.diagflat(output) - output@output.T)@grad_i

        self.dinputs = grad_inputs

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


class Activation_Linear():

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues

    def predictions(self, outputs):
        return outputs


class Loss():

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, include_regularization=True):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_count += len(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        if include_regularization:
            return data_loss, self.regularization_loss()
        return data_loss

    def calculate_accumulated(self, *, include_regularization=True):
        if include_regularization:
            return self.accumulated_sum/self.accumulated_count, self.regularization_loss()
        return self.accumulated_sum/self.accumulated_count

    def regularization_loss(self):

        regularization_loss = 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights**2)

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases**2)

        return regularization_loss


class Loss_MeanSquaredError(Loss):

    def forward(self, y_pred, y_true):
        return np.mean((y_pred-y_true)**2, axis=1)

    def backward(self, y_pred, y_true):
        self.dinputs = 2*(y_pred-y_true)/(len(y_pred)*len(y_pred[0]))


class Loss_MeanAbsoluteError(Loss):

    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_pred-y_true), axis=1)

    def backward(self, y_pred, y_true):
        self.dinputs = np.ones_like(y_pred)
        self.dinputs[y_pred-y_true < 0] = -1
        self.dinputs /= len(y_pred)


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            return -np.log(y_pred_clipped[range(len(y_pred_clipped)), y_true])

        elif len(y_true.shape) == 2:
            return -np.log(np.sum(y_pred_clipped*y_true, axis=1))

    def backward(self, y_pred, y_true):
        n_samples = len(y_pred)
        n_labels = len(y_pred[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        self.dinputs = -y_true/(y_pred*n_samples)


class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        return np.mean(-(y_true*np.log(y_pred_clipped)+(1-y_true)*np.log(1-y_pred_clipped)), axis=1)

    def backward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        self.dinputs = (-(y_true/y_pred_clipped-(1-y_true) /
                        (1-y_pred_clipped)))/(len(y_pred[0])*len(y_pred))


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.activation.output, y_true)

    def backward(self, outputs, y_true):
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = outputs.copy()
        self.dinputs[range(len(y_true)), y_true] -= 1
        self.dinputs /= len(y_true)


class Optimizer_SGD():

    def __init__(self, learning_rate=1, decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1+self.iterations * self.decay)

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_cache"):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_cache - \
                self.current_learning_rate * layer.dweights
            bias_updates = self.momentum * layer.bias_cache - \
                self.current_learning_rate * layer.dbiases
            layer.weight_cache = weight_updates
            layer.bias_cache = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad():

    def __init__(self, learning_rate=1, decay=0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1+self.iterations * self.decay)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
            layer.dweights / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop():

    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1+self.iterations * self.decay)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho*layer.weight_cache + \
            (1-self.rho)*layer.dweights ** 2
        layer.bias_cache = self.rho*layer.bias_cache + \
            (1-self.rho)*layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * \
            layer.dweights / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam():

    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / \
                (1+self.iterations * self.decay)

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + (1-self.beta_1)*layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + (1-self.beta_1)*layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1-self.beta_1**(1+self.iterations))
        bias_momentums_corrected = layer.bias_momentums / \
            (1-self.beta_1**(1+self.iterations))

        layer.weight_cache = self.beta_2*layer.weight_cache + \
            (1-self.beta_2)*layer.dweights ** 2
        layer.bias_cache = self.beta_2*layer.bias_cache + \
            (1-self.beta_2)*layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / \
            (1-self.beta_2**(1+self.iterations))
        bias_cache_corrected = layer.bias_cache / \
            (1-self.beta_2**(1+self.iterations))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Layer_Dropout():

    def __init__(self, rate):
        self.rate = 1-rate

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if training:
            self.binary_mask = np.random.binomial(
                1, self.rate, size=inputs.shape)/self.rate
            self.output = self.binary_mask * inputs
        else:
            self.output = inputs

    def backward(self, dvalues):
        self.dinputs = self.binary_mask*dvalues


class Layer_Input():

    def forward(self, inputs, training=False):
        self.output = inputs


class NeuralNetworkModel():

    def __init__(self, layers_sizes, optimizer, reg_loss=None):
        if reg_loss == None:
            self.layers = [Layer_Dense(layers_sizes[i], layers_sizes[i+1])
                           for i in range(len(layers_sizes)-1)]
        else:
            self.layers = [Layer_Dense(
                layers_sizes[i], layers_sizes[i+1], *reg_loss[i]) for i in range(len(layers_sizes)-1)]
        self.relu = [Activation_ReLU() for i in range(len(layers_sizes)-2)]
        self.softmax = Activation_Softmax_Loss_CategoricalCrossentropy()
        self.optimizer = optimizer

    def forward(self, inputs, y=None, training=False):
        self.layers[0].forward(inputs)
        for i in range(len(self.layers)-1):
            self.relu[i].forward(self.layers[i].output)
            self.layers[i+1].forward(self.relu[i].output)
        if y is None:
            self.softmax.activation.forward(self.layers[-1].output)
            return self.softmax.activation.output
        return self.softmax.forward(self.layers[-1].output, y)

    def backward(self, outputs, y_true):
        self.softmax.backward(outputs, y_true)
        self.layers[-1].backward(self.softmax.dinputs)
        for i in range(len(self.layers)-2, -1, -1):
            self.relu[i].backward(self.layers[i+1].dinputs)
            self.layers[i].backward(self.relu[i].dinputs)

    def train_batch(self, X, y):
        self.forward(X, y)
        self.backward(self.softmax.output, y)
        self.optimizer.pre_update_params()
        for layer in self.layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update_params()

    def accuracy(self, X_test, y_true):
        loss = self.forward(X_test, y_true)
        predictions = np.argmax(self.softmax.output, axis=1)
        return np.mean(predictions == y_true), loss+np.sum([self.softmax.loss.regularization_loss(layer) for layer in self.layers])


class Model():

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, batch_size=None, epochs=1, print_every=1, validation_data=None):

        train_steps = 1

        if batch_size is not None:
            train_steps = len(X) // batch_size
            if batch_size*train_steps < len(X):
                train_steps += 1

        self.accuracy.init(y, reinit=True)
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                if batch_size:
                    batch_X=X[step*batch_size:(step+1)*batch_size]
                    batch_y =y[step*batch_size:(step+1)*batch_size]
                else:
                    batch_X = X
                    batch_y = y
                output = self.forward(batch_X, True)
                data_loss, reg_loss = self.loss.calculate(output, batch_y)
                loss = data_loss + reg_loss
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                self.backward(output, batch_y)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                
                if step % print_every == 0 or step==train_steps-1:
                    print(f'step:{step}, acc: {accuracy:.3f}, loss:{loss:.3f}, (data loss:{data_loss:.3f}, reg_loss:{reg_loss:.3f}), lr:{self.optimizer.current_learning_rate}')
            accuracy = self.accuracy.calculate_accumulated()
            data_loss, reg_loss = self.loss.calculate_accumulated()
            loss = data_loss+reg_loss
            print(f'training, acc: {accuracy:.3f}, loss:{loss:.3f}, (data loss:{data_loss:.3f}, reg_loss:{reg_loss:.3f}), lr:{self.optimizer.current_learning_rate}')
            
            
        if validation_data:
            self.evaluate(*validation_data, batch_size=batch_size)
            
    def finalize(self):
        self.input_layer = Layer_Input()
        layer_count = len(self.layers)
        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count-1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        self.output_layer_activation = self.layers[-1]
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.output_layer_activation, Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, inputs, training):
        self.input_layer.forward(inputs, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y_true):
        if self.softmax_classifier_output is None:
            self.loss.backward(output, y_true)
            for layer in self.layers[::-1]:
                layer.backward(layer.next.dinputs)
        else:
            self.softmax_classifier_output.backward(output, y_true)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            for layer in self.layers[-2::-1]:
                layer.backward(layer.next.dinputs)
                
    def evaluate(self, X_val, y_val, *, batch_size = None):
        
        self.loss.new_pass()
        self.accuracy.new_pass()
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val)//batch_size
            if batch_size*validation_steps < len(X_val):
                validation_steps += 1
        for step in range(validation_steps):
            if batch_size:
                batch_X=X_val[step*batch_size:(step+1)*batch_size]
                batch_y =y_val[step*batch_size:(step+1)*batch_size]
            else:
                batch_X = X_val
                batch_y = y_val
            output = self.forward(batch_X, False)
            data_loss, reg_loss = self.loss.calculate(output, batch_y)
            loss = data_loss + reg_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
        data_loss, reg_loss = self.loss.calculate_accumulated()
        loss = data_loss + reg_loss
        accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, acc: {accuracy:.3f}, loss:{loss:.3f}')

    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters
    
    def set_parameters(self, parameters):
        for layer, (weights, biases) in zip(self.trainable_layers, parameters):
            layer.set_parameters(weights, biases)

    def save_parameters(self, name):
        with open(name, '+bw') as file:
            pickle.dump(self.get_parameters(), file)
            
    def load_parameters(self, name):
        with open(name, 'rb') as file:
            parameters = pickle.load(file)
        self.set_parameters(parameters)
        
    def save(self, path):
        
        model = copy.deepcopy(self)
        model.accuracy.new_pass()
        model.loss.new_pass()
        model.input_layer.__dict__.pop("output", None)
        model.loss.__dict__.pop("dinputs", None)
        for layer in model.trainable_layers:
            for property in ['dweights', 'dbiases', 'dinputs', 'output', 'inputs']:
                layer.__dict__.pop(property, None)
        
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    
    def predict(self, X, *, batch_size=None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            if batch_size*prediction_steps < len(X):
                prediction_steps += 1
        outputs = []
        for step in range(prediction_steps):
                if batch_size:
                    batch_X=X[step*batch_size:(step+1)*batch_size]
                else:
                    batch_X = X
                output = self.forward(batch_X, True)                        
                outputs.append(output)

        return np.vstack(outputs)
    
    @staticmethod
    def load(path):
        
        with open(path, 'rb') as file:
            return pickle.load(file)
    
class Accuracy():

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_count+=len(comparisons)
        self.accumulated_sum+=np.sum(comparisons)
        return accuracy

    def calculate_accumulated(self):
        return self.accumulated_sum/self.accumulated_count

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y)/250

    def compare(self, predictions, y):
        return np.abs(predictions-y) < self.precision


class Accuracy_Categorical(Accuracy):

    def __init__(self, binary):
        self.binary = binary

    def init(self, y, reinit=False):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2 and not self.binary:
            y = np.argmax(y, axis=1)
        return predictions == y

