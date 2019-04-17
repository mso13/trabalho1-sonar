import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    
    # Constructor
    
    def __init__(self, n_inputs, lRate):
        self.lRate = lRate
        self.weights = np.random.rand(n_inputs + 1) * 0.2 - 0.1 # Num. features + 1 (bias)
    
    # Prediction

    def predict(self, inputs):
        activation = np.dot(np.append(inputs, 1.0), self.weights)
        return 1.0 if activation >= 0.0 else 0.0

    # Learn

    def learn(self, trainingSet_inputs, trainingSet_outputs, nEpoch):
        #weights = np.random.rand(61) * 0.2 - 0.1 # Num. features + 1 (bias)
        error_list = []
        accuracy_list = []
        for epoch in range(nEpoch):
            sumError = 0.0
            for inputs, output in zip(trainingSet_inputs, trainingSet_outputs) :
                prediction = self.predict(inputs)
                error = output - prediction
                sumError += error ** 2
                self.weights = self.weights + self.lRate * error * np.append(inputs, 1.0)
            error_list.append(sumError)
            accuracy = 100*(len(trainingSet_outputs)-sumError)/len(trainingSet_outputs)
            accuracy_list.append(accuracy)
            if epoch % 1000 == 0:
                print ('Epoch %d Error: %.3f Accuracy: %.3f (%%)' % (epoch, sumError, accuracy))
            
        plt.plot(error_list)
        plt.title('Quadratic Error x Epoch')
        plt.ylabel('Quadratic Error')
        plt.xlabel('Epoch')
        fig = plt.gcf() # stores figure to save it
        fig.savefig('error.png')
        plt.show()
        
        plt.plot(accuracy_list)
        plt.title('Accuracy (%) x Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        fig = plt.gcf() # stores figure to save it
        fig.savefig('accuracy.png')
        plt.show()
        
        return self.weights