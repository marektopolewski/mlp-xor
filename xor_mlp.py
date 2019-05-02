from numpy import exp, array, random, dot, ones, mean
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import time

# Methods for importing training and test datasets
def noisy_test_data(input_fields, output_field, data_size=64):
    data = pd.read_csv('noisy_data.csv')
    x = array(data[input_fields])
    t = array([data[output_field]]).T
    return x[len(x)-64:], t[len(x)-64:]

def fake_train_data(input_fields, output_field):
    x_train = array([[-0.310691, -0.309003, 1.25774, 1.31959, -0.0897083, -0.457115, 1.42524, 1.43962, 
        -0.21377, -0.16744, 0.579612, 1.90558, 0.442017, 0.204012, 1.75664, 0.584128],
        [0.0164278, 0.898471, -0.231735, 0.82952, -1.02045, 1.84369, 0.111823, 0.28365, 
        0.0759174, 0.985518, 0.584378, 0.434351, 0.35245, -0.0194183, -0.336488, 1.45608], 
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    
    t_train = array([[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]])
    
    return x_train.T, t_train.T

def generate_noisy_data(input_fields, output_field, data_size, seed_flag, export_flag=False):

    data = pd.read_csv('clean_data.csv')

    # add noise to the data
    if seed_flag:
        random.seed(8)

    variance = 0.25
    mu, sigma = 0, sqrt(variance)
    a_noise = random.normal(mu, sigma, [len(data.index)]) 
    b_noise = random.normal(mu, sigma, [len(data.index)]) 
    data[input_fields[0]] += a_noise
    data[input_fields[1]] += b_noise

    # export the noisy data to a CSV
    if export_flag:
        df_noisy = pd.DataFrame()
        df_noisy["a"] = data[input_fields[0]]
        df_noisy["b"] = data[input_fields[1]]
        df_noisy["offset"] = -1
        df_noisy["c"] = data[output_field]
        df_noisy.to_csv("new_noisy_data.csv", header=True, index=False)

    # separate data into a feature matrix and true outcome vector
    x_train, x_test = array(data[input_fields][:data_size]), array(data[input_fields][len(data)-64:])
    t_train, t_test = array(data[output_field][:data_size]), array(data[output_field][len(data)-64:])

    x_train_fake, t_train_fake = fake_train_data(input_fields, output_field)
    x_train[:len(x_train_fake)] = x_train_fake
    t_train[:len(t_train_fake)] = t_train_fake.T

    return x_train, array([t_train]).T, x_test, array([t_test]).T

def import_noisy_data(input_fields, output_field, data_size, control_flag=True):
    data = pd.read_csv('noisy_data.csv')

    x = array(data[input_fields])
    t = array([data[output_field]]).T

    x_train, t_train = x[:data_size], t[:data_size]
    x_test, t_test = noisy_test_data(input_fields, output_field)

    if control_flag:
        x_train_fake, t_train_fake = fake_train_data(input_fields, output_field)
        x_train[:len(x_train_fake)] = x_train_fake
        t_train[:len(t_train_fake)] = t_train_fake

    return x_train, t_train, x_test, t_test


# Structure for a single neuron layer storing its weight matrix
class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron, bias_flag):
        if bias_flag:
            self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron+1, number_of_neurons)) - 1
        else:
            self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


# Structure for a 3-layer perceptron using SGD and sigmoid as the activation function
class NeuralNetwork():
    def __init__(self, params, layer1, layer2):
        self.params = params
        self.layer1 = layer1
        self.layer2 = layer2
        self.errors = [None]

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __add_bias(self, x, val):
        x_with_bias = val * ones((len(x),len(x[0])+1))
        x_with_bias[:,:-1] = x
        return x_with_bias

    def train(self, x_train, t_train, number_of_training_iterations):
        self.errors = [None] * number_of_training_iterations
        for iteration in range(number_of_training_iterations):
            temp_errors = [None] * len(x_train)
            
            for idx in range(len(x_train)):
                x, t = array([x_train[idx]]), array([t_train[idx]])
                # Pass the training set through our neural network
                output_from_layer_1, output_from_layer_2 = self.think(x)

                # Calculate the error for layer 2 (difference between the desired and the predicted output).
                layer2_error = t - output_from_layer_2
                layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
                temp_errors[idx] = (layer2_error[0])**2

                # Calculate the error for layer 1 (By looking at the weights in layer 1,
                # we can determine by how much layer 1 contributed to the error in layer 2).
                layer1_error = layer2_delta.dot(self.layer2.synaptic_weights[:-1].T)
                layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

                # Calculate how much to adjust the weights by
                layer1_adjustment = x.T.dot(layer1_delta)
                layer2_adjustment = self.__add_bias(output_from_layer_1,-1).T.dot(layer2_delta)

                # Adjust the weights
                learing_rate = 1.1
                self.layer1.synaptic_weights += learing_rate * layer1_adjustment
                self.layer2.synaptic_weights += learing_rate * layer2_adjustment

            self.errors[iteration] = mean(temp_errors)


    def train_batch(self, x_train, t_train, number_of_training_iterations):
        self.errors = [None] * int(number_of_training_iterations)
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(x_train)

            # Calculate the error for layer 2 (difference between the desired and the predicted output).
            layer2_error = t_train - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
            self.errors[iteration] = mean((layer2_error)**2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights[:-1].T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = x_train.T.dot(layer1_delta)
            layer2_adjustment = self.__add_bias(output_from_layer_1,-1).T.dot(layer2_delta)

            # Adjust the weights
            learing_rate = 1
            self.layer1.synaptic_weights += learing_rate * layer1_adjustment
            self.layer2.synaptic_weights += learing_rate * layer2_adjustment
           

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        out_l1_with_bias = self.__add_bias(output_from_layer1, -1)
        output_from_layer2 = self.__sigmoid(dot(out_l1_with_bias, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 - %i neuron(s), each with %i input(s)" 
            % (self.layer1.synaptic_weights.shape[1], self.layer1.synaptic_weights.shape[0]))
        print(self.layer1.synaptic_weights)
        print("    Layer 2 - %i neuron(s), each with %i input(s)" 
            % (self.layer2.synaptic_weights.shape[1], self.layer2.synaptic_weights.shape[0]))
        print(self.layer2.synaptic_weights)

    def plot_classification(self):
        res = 49
        output = [None] * (res+1)

        for i in range(res+1):
            output[i] = [None] * (res+1)
            for j in range(res+1):
                x = array([float(i)/res, float(j)/res, -1]).reshape(1, -1)
                hidden_state, prediction = self.think(x)
                output[i][j] = prediction[0][0]

        fig = plt.figure(figsize=(12,4))
        fig.suptitle(
            "# hidden neurons="+str(self.params['neurons'])+", data set size="+str(self.params['data_size']), 
            fontsize=14 )
        ax = plt.subplot(121)
        cax = plt.imshow(array(output), interpolation='nearest', vmin=0, vmax=1)
        plt.set_cmap('gray')
        plt.axis('off')
        res = res+1
        table = {'0, 0':(-2, -1), '0, 1':(-2, res+2), '1, 0':(res-2, -1), '1, 1':(res-2, res+2)}
        for text, corner in table.items():
            ax.annotate(text, xy=corner, size=12, annotation_clip=False)

        ax = plt.subplot(122)
        plt.plot(self.errors)
        ax.set_ylim(bottom=0, top=0.51)
        plt.draw()


def test_mlp(hidden_layer_neurons, data_size, dataFlag, batchFlag, seedFlag, verbose):

    ################################ IMPORT TRAINING DATA ################################
    input_fields, output_field = ['a','b','offset'], 'c'

    if dataFlag == 0:
        x_train, t_train, x_test, t_test = array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1]]), array([[0],[1],[1],[0]]), array([[]]), array([[]])
    elif dataFlag == 1:
        x_train, t_train, x_test, t_test = generate_noisy_data(input_fields, output_field, data_size, seedFlag)
    else:
        x_train, t_train, x_test, t_test = import_noisy_data(input_fields, output_field, data_size, True)


    ################################ DEFINE THE MLP MODEL ################################
    if batchFlag and seeded:
        random.seed(20)
    elif seeded:
        random.seed(3)
   
    layer1 = NeuronLayer(hidden_layer_neurons, len(input_fields), False)
    layer2 = NeuronLayer(1, hidden_layer_neurons, True)

    # Combine the layers to create a neural network
    params = {'neurons':hidden_layer_neurons, 'data_size':data_size}
    neural_network = NeuralNetwork(params, layer1, layer2)

    if verbose > 2:
        print("Initial (random) network weights: ")
        neural_network.print_weights()


    ################################ TRAIN THE MLP MODEL ################################
    epochs = int(16000 / x_train.shape[0])
    if batchFlag:
        neural_network.train_batch(x_train, t_train, epochs)
    else:
        neural_network.train(x_train, t_train, epochs)

    if verbose > 2:    
        print("\nWeights after training:")
        neural_network.print_weights()

    ################################## PLOT OUTPUT MAP ##################################
    if verbose > 1: 
        neural_network.plot_classification()

    ############################ CALCULATE MSE ON TEST SET ##############################
    hidden_state, t_pred = neural_network.think(x_test)
    mse = ((t_pred - t_test)**2).mean(axis=None)
    if verbose > 0:
        print("neurons=%i, data size=%i, epochs=%i, MSE=%.6f" 
            % (hidden_layer_neurons, data_size, epochs, mse))

    return mse


# Main method
if __name__ == "__main__":

    #====#====# Use these paraeters to adjust testing configs #====#====#====#====#====#====#
    verbose = 2                 # 0 -> no info, 3 -> all info
    batch = False               # True -> batch mode, False -> online mode
    seeded = True               # True -> init is seeded to match those in report
    dataMode = 2                # 0 -> noiseless, 1 -> create new noisy, 2 -> like report
    neuron_nums = [2, 4, 8]
    data_sizes = [16, 32, 64]
    #====#====#====#====#====#====#====#====#====#====#====#====#===#===#====#====#====#====#

    colour_map = { 0:'r', 1:'g', 2:'b' }
    mse = ones([len(neuron_nums), len(data_sizes)])

    for i in range(len(neuron_nums)):
        for j in range(len(data_sizes)):
            mse[i][j] = test_mlp(neuron_nums[i], data_sizes[j], dataMode, batch, seeded, verbose)

    plt.figure(figsize=(6,6))
    ax = plt.gca()
    plt.grid()
    for i in range(len(neuron_nums)):
        plt.scatter(data_sizes, mse[i], c=colour_map[i%len(colour_map)])
        plt.plot(data_sizes, mse[i], c=colour_map[i%len(colour_map)], label=""+str(neuron_nums[i])+" neurons")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), shadow=True, ncol=3)
    plt.xlabel('# of training samples', fontsize=12)
    plt.ylabel('mean squared error', fontsize=12)
    ax.set_ylim(bottom=0, top=0.28)
    plt.draw()

    plt.show()

    # run a script to test best model on average
    print("\nCalculating the best model in terms of average MSE...\n")
    print("Neurons \t Data size \t Batch mode \t MSE")
    iters = 50
    times = []
    for batchFlag in [True,False]:
        start = time.time()
        for i in range(len(neuron_nums)):
            for j in range(len(data_sizes)):
                mse = 0
                for it in range(iters):
                    mse += test_mlp(neuron_nums[i], data_sizes[j], 1, batchFlag, False, 0)
                mse = mse / iters
                print("%i \t\t %i \t\t %s \t\t %.6f" % (neuron_nums[i], data_sizes[j], batchFlag, mse))
        end = time.time()
        times.append((end-start)/iters)

    print("\nTime to exectute:\n\tBatch = %.6f\tOnline = %.6f" % (times[0],times[1]))
    print("\nEND OF SCRIPT\n")