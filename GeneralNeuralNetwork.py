import numpy as np
from sklearn import datasets
import numpy as np
# XOR dataset
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[0,1,1,0]]).T

#random initialization of weights
weights_Hidden=2*np.random.random((2,2))-1
bias_Hidden=2*np.random.random((1,2))-1
weights_Output=2*np.random.random((2,1))-1
bias_Output=2*np.random.random((1,1))-1
lr=0.1

def sig(z):
    return 1/(1+np.exp(-z))
def derivative_sig(z):
    return sig(z)*(1-sig(z))

# training weights and bias
for i in range(10000):   
    output0 = X
    input_HiddenLayer = np.dot(output0,weights_Hidden)+bias_Hidden
    outputHidden = sig(input_HiddenLayer)
    input_OutputLayer = np.dot(outputHidden,weights_Output)+bias_Output
    output = sig(input_OutputLayer)
    
    first_term_output_Layer = output-Y
    second_term_output_Layer = derivative_sig(input_OutputLayer)
    first_two_term_Output_Layer = first_term_output_Layer*second_term_output_Layer
    
    changes_output = np.dot(outputHidden.T,first_two_term_Output_Layer)
    changes_output_bias = np.sum (first_two_term_Output_Layer,axis=0,keepdims = True)
    
    first_term_hidden_Layer = np.dot(first_two_term_Output_Layer,weights_Output.T)
    second_term_hidden_Layer = derivative_sig(input_HiddenLayer)
    first_two_term_Hidden_Layer = first_term_hidden_Layer*second_term_hidden_Layer
    
    changes_Hidden = np.dot(output0.T,first_two_term_Hidden_Layer)
    changes_hidden_bias = np.sum (first_two_term_Hidden_Layer,axis=0,keepdims = True)
    
    weights_Hidden = weights_Hidden - lr*changes_Hidden
    bias_Hidden = (bias_Hidden - lr*changes_hidden_bias)
    weights_Output = (weights_Output - lr*changes_output)
    bias_Output = (bias_Output - lr*changes_output_bias)

    

#predict output
input_HiddenLayer = np.dot(output0,weights_Hidden)+bias_Hidden
outputHidden = sig(input_HiddenLayer)
input_OutputLayer = np.dot(outputHidden,weights_Output)+bias_Output
output = sig(input_OutputLayer)
print(output)  
