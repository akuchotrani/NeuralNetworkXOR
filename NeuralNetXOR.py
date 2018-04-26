# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:06:45 2018

@author: aakash.chotrani
"""

import numpy as np

#################################################################################

def initialize_data():
    global XORdata
    global X
    global y
    #Xor data
    XORdata=np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
    X=XORdata[:,0:2]
    y=XORdata[:,-1]
    



#################################################################################


def print_network(net):
    for i,layer in enumerate(net,1):
        print("Layer {} ".format(i))
        for j,neuron in enumerate(layer,1):
            print("neuron {} :".format(j),neuron)
    print('#################################################################################')


#################################################################################

def initialize_network():
    
    input_neurons=len(X[0])
    hidden_neurons=input_neurons+1
    output_neurons=2
    
    n_hidden_layers=1
    
    net=list()
    
    for h in range(n_hidden_layers):
        if h!=0:
            input_neurons=len(net[-1])
            
        hidden_layer = [ { 'weights': np.random.uniform(size=input_neurons)} for i in range(hidden_neurons) ]
        net.append(hidden_layer)
    
    output_layer = [ { 'weights': np.random.uniform(size=hidden_neurons)} for i in range(output_neurons)]
    net.append(output_layer)
    
    return net


#################################################################################

def  activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))


#################################################################################
    
def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            sum=neuron['weights'].T.dot(row)
            
            result=activate_sigmoid(sum)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input
    
    return row



#################################################################################
def sigmoidDerivative(output):
    return output*(1.0-output)

#################################################################################
    
def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results) 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
                
 
################################################################################# 
                
def updateWeights(net,input,lrate):
    
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]


#################################################################################                
def training(net, epochs,lrate,n_outputs):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        for i,row in enumerate(X):
            outputs=forward_propagation(net,row)
            
            expected=[0.0 for i in range(n_outputs)]
            expected[y[i]]=1
    
            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(len(expected))])
            back_propagation(net,row,expected)
            updateWeights(net,row,0.05)
        if epoch%1000 ==0:
            print('Current EPOCH =%d ERROR=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors




#################################################################################
#Prediction after weights are updated. Doing one step of forward propogation. Now the network will spit 
def predict(network, row):
    outputs = forward_propagation(network, row)
    return outputs



#################################################################################
#################################################################################
def main():
    
    
    initialize_data()
    net=initialize_network()
    
    print('PRINTING NETWORK')
    print_network(net)
    
    
    print('TRAINING NETWORK')
    #training will take input
    #1)trained neural network
    #2)total epoch.
    #3)learning rate
    #4)number of outputs.
    training(net,20000, 0.05,2)
    




    print('#################################################################################')
    print('PREDICTION')
    input1 = np.array([0,0])
    input2 = np.array([0,1])
    input3 = np.array([1,0])
    input4 = np.array([1,1])
    
    
    pred=predict(net,input1)
    output=np.argmax(pred)
    print('Input: ',input1,' Output: ',output)
    
    pred=predict(net,input2)
    output=np.argmax(pred)
    print('Input: ',input2,' Output: ',output)
    
    pred=predict(net,input3)
    output=np.argmax(pred)
    print('Input: ',input3,' Output: ',output)
    
    pred=predict(net,input4)
    output=np.argmax(pred)
    print('Input: ',input4,' Output: ',output)
    print('#################################################################################')

    

if __name__ == "__main__":
    main()


