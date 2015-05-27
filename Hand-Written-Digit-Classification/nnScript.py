import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time

def encode(A):
    
    row= np.repeat(0,10)
    mat=row[np.newaxis,:]
    for x in np.nditer(A):
        row=np.repeat(0,10)
        row[x]=1
        concat = row[np.newaxis,:]
        mat=np.concatenate((mat,concat),0)
    mat=mat[1:,:]
    return mat



def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1 / (1 + np.exp(-z))#your code here

def preprocess():
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    train_data_temp = np.concatenate((mat['train0'],mat['train1'],mat['train2'],mat['train3'],mat['train4'],mat['train5'],mat['train6'],mat['train7'],mat['train8'],mat['train9']),axis=0)
    train_data = np.array(train_data_temp)
    features = np.size(train_data,1)
    train_data_size=50000
    label0 = np.tile([0,0,0,0,0,0,0,0,0,0],(np.size(mat['train0'],0),1))
    train_label_temp = label0
    for x in range(1,10):
        row= np.repeat(0,10)
        row[x]=1
        train_label_temp = np.concatenate((train_label_temp,np.tile(row,(np.size(mat['train'+str(x)],0),1))),axis=0)

       #print train_label_temp.shape
    train_label = np.array(train_label_temp)
    trainval_data = np.concatenate((train_data,train_label),axis=1)
    trainval_data = np.double(trainval_data)
    #Randomize the rows of the training
    np.random.shuffle(trainval_data)
    #Slice the training data and the training label
    train_data = trainval_data[0:train_data_size,0:features]
    train_label = trainval_data[0:train_data_size,features:]
    #Slice the validation data and the validation label
    validation_data = trainval_data[train_data_size:,0:features]
    validation_label = trainval_data[train_data_size:,features:]
    test_data_temp = np.concatenate((mat['test0'],mat['test1'],mat['test2'],mat['test3'],mat['test4'],mat['test5'],mat['test6'],mat['test7'],mat['test8'],mat['test9']),axis=0)
    test_data = np.array(test_data_temp)
    test_data = np.double(test_data)

    test_label0 = np.tile([0,0,0,0,0,0,0,0,0,0],(np.size(mat['test0'],0),1))
    test_label_temp = test_label0
    for x in range(1,10):
        row= np.repeat(0,10)
        row[x]=1
        test_label_temp = np.concatenate((test_label_temp,np.tile(row,(np.size(mat['test'+str(x)],0),1))),axis=0)
    test_label = np.array(test_label_temp)
    
    test_label=np.argmax(test_label,axis=1)
    validation_label=np.argmax(validation_label,axis=1)
    train_label=np.argmax(train_label,axis=1)
    train_data=train_data/255
    validation_data=validation_data/255
    test_data=test_data/255
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""
     
    labels = np.array([])
    
    # create bias row
    bias_row =np.ones((np.size(data,0),1))
    
    # concatenate bias with data matrix
    data=np.concatenate((data,bias_row),axis=1)
    
    #Calculate input to hidden layer
    intput_hidden_layer= np.dot(data,w1.transpose())                                
    
    #Calculate output of hidden layer using sigmoid function
    output_hidden_layer= sigmoid(intput_hidden_layer)
    
    #Calculate input to output nodes
    input_with_bias = np.concatenate((output_hidden_layer,bias_row),axis=1)                                                                                             
    input_output_node= np.dot(input_with_bias,w2.transpose()) 
    
    # Calculate output of output layer
    output_layer= sigmoid(input_output_node)  
    
    # get index of maximum from all rows in ouput layer matrix
    labels = np.argmax(output_layer,axis=1)  
    
    return labels

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    obj_grad = np.array([])
    
            
      
    # initialize a  gradient vector matrix with w1
    gradient_matrix_w1 = np.zeros(w1.shape)   
    
    # initialize a gradient vector matrix with w2
    gradient_matrix_w2 = np.zeros(w2.shape)   
    
    #transpose of input matrix
    input_mat = training_data.transpose() 
    
    #transpose of label matrix(expected output)
    label_expected_mat = training_label.transpose() 
    
    #get number of coulumns in matrix a1 which is equal to the total number of inputs
    number_of_input_columns = input_mat.shape[1]     
    
    #Calculate the bias as input from hiddenlayer
    input_bias = np.tile(1,(1,np.size(input_mat,1)))
    
    #adding bias column in matrix
    input_mat=np.concatenate((input_mat,input_bias), axis=0)       
    
    # multipy input with the corresponding wieght vector
    input_hidden_layer = np.dot(w1,input_mat)
    
    # output of hidden layer  
    output_hidden_layer = sigmoid(input_hidden_layer)  
    
    # calculate bias for hidden layer
    hidden_layer_bias = np.tile(1,(1,np.size(output_hidden_layer,1)))
    
    #adding bias column in hidden layer
    output_hidden_layer = np.concatenate((output_hidden_layer,hidden_layer_bias), axis=0) 
    
    # multipy output of hidden layer with the corresponding wieght vector
    input_output_layer = np.dot(w2,output_hidden_layer)                                            
    
    # output of output layer
    output_output_layer = sigmoid(input_output_layer)     
    
    # Regularization of neural network
    error_function_mat=label_expected_mat*np.log(output_output_layer)+(1-label_expected_mat)*np.log(1-output_output_layer) 
    obj_val= - np.sum(error_function_mat[:])/number_of_input_columns
    
    #ew1=w1**2
    #ew2=w2**2
    error_function = ((np.sum(pow(w1[:],2)) + np.sum(pow(w2[:],2)))/(2*number_of_input_columns))*lambdaval
    obj_val=obj_val+error_function
    
    print obj_val
    # claculate delta for hidden and output layer 
    delta_output_layer = (output_output_layer - label_expected_mat)  
    delta_hidden_layer =  np.dot(w2.transpose(), delta_output_layer)*(output_hidden_layer*(1-output_hidden_layer))
    change_in_W1 = np.dot(delta_hidden_layer , input_mat.transpose())
    change_in_W2 = np.dot(delta_output_layer , output_hidden_layer.transpose())
    
    #remove the bias row
    change_in_W1 = change_in_W1[:-1,:]
    
    
    #update weight matrix
    gradient_matrix_w1 = gradient_matrix_w1 + change_in_W1
    gradient_matrix_w2 = gradient_matrix_w2 + change_in_W2
    
    #regularization of updated weight matrix
    gradient_matrix_w1 = (gradient_matrix_w1 + lambdaval*w1)/number_of_input_columns
    gradient_matrix_w2 = (gradient_matrix_w2 + lambdaval*w2)/number_of_input_columns
    obj_grad = np.concatenate((gradient_matrix_w1.flatten(),gradient_matrix_w2.flatten()), axis=0)    
        
    print obj_grad
    return (obj_val,obj_grad)        

                
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50
	    				    				    				   
# set the number of nodes in output unit
n_class = 10				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.7

train_label=encode(train_label)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
train_label=np.argmax(train_label,axis=1)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


predicted_label = nnPredict(w1,w2,train_data)
                                                                                                                        
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')