""" This is the class for a 2 hidden layer neural-network. The structure of
the classifier follows closely the structure of classifiers in Scikit-Learn, as in there
are there methods:  1- The constructor initializes the weights, 2- fit trains the neural
networks, 3- predict_proba output the predicted probabilities. """

from numpy import *

class mlp2(object):
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic', C=1.0):
        """ Constructor """
        # Set up network size
        self.nin = shape(inputs)[1]
        self.nout = shape(targets)[1]
        self.ndata = shape(inputs)[0]
        self.nhidden = nhidden  # number of hidden units for layer 1
	self.nhidden2 = nhidden	 # number of hidden units for layer 2

        self.beta = beta  #coefficient of the exponential in logistic regression
        self.momentum = momentum
        self.outtype = outtype
	self.C = C  # resularization parameter
    
        # Initialise network
        self.weights1 = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
	self.weights12 = (random.rand(self.nhidden+1,self.nhidden2)-0.5)*2/sqrt(self.nhidden)
        self.weights2 = (random.rand(self.nhidden2+1,self.nout)-0.5)*2/sqrt(self.nhidden2)

	
    def fit(self,inputs,targets,weightsdata,eta,niterations,decaybase):  
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
	
	self.weights=weightsdata
	self.sum=sum(self.weights)
    
        updatew1 = zeros((shape(self.weights1)))
	updatew12 = zeros((shape(self.weights12)))
        updatew2 = zeros((shape(self.weights2)))
                      
        for n in range(niterations):
    
            self.outputs = self.predict_proba(inputs,fit=True)

	    if self.outtype == 'logistic':
		error = -sum(((targets)*log(self.outputs)+(1.0-targets)*log(1.0-self.outputs))*self.weights)/self.sum
	    else:
		error = sum((targets-self.outputs)**2*self.weights)/self.sum
		
	    
            if (mod(n,50)==0):
                print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (targets-self.outputs)/self.ndata*self.weights
            elif self.outtype == 'logistic':
            	deltao = (targets-self.outputs)*self.outputs*(1.0-self.outputs)*self.weights
            elif self.outtype == 'softmax':
            	#deltao = (targets-self.outputs)*self.outputs/self.ndata
                deltao = (targets-self.outputs)/self.ndata*self.weights
            else:
            	print "error"
            
	    
	    deltah2 = self.hidden2*(1.0-self.hidden2)*((dot(deltao,transpose(self.weights2))))
	    deltah=self.hidden*(1.0-self.hidden)*((dot(deltah2[:,:-1],transpose(self.weights12))))
	    
            updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
	    updatew12= eta*(dot(transpose(self.hidden),deltah2[:,:-1])) + self.momentum*updatew12
            updatew2 = eta*(dot(transpose(self.hidden2),deltao)) + self.momentum*updatew2
	    	    
            self.weights1 += updatew1
	    self.weights12 += updatew12
            self.weights2 += updatew2
	    
	    #Renormalization
	    norm1=sqrt(sum(self.weights1**2,axis=0))
	    norm12=sqrt(sum(self.weights12**2,axis=0))
	    norm2=sqrt(sum(self.weights2**2,axis=0))
	    
	    norma1=ones(shape(norm1))+(norm1>self.C).astype(float)*(norm1/self.C-ones(shape(norm1)))
	    norma12=ones(shape(norm12))+(norm12>self.C).astype(float)*(norm12/self.C-ones(shape(norm12)))
	    norma2=ones(shape(norm2))+(norm2>self.C).astype(float)*(norm2/self.C-ones(shape(norm2)))
	    
	    self.weights1=self.weights1/norma1
	    self.weights12=self.weights12/norma12
	    self.weights2=self.weights2/norma2
                
            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]
	    self.weights = self.weights[change,:]
	    eta=eta*decaybase
            
    def predict_proba(self,inputs,fit=False):
	
	if not fit:
	    inputs = concatenate((inputs,-ones((len(inputs),1))),axis=1)
	
        self.hidden = dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+exp(-self.beta*self.hidden))
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)
	
	self.hidden2 = dot(self.hidden,self.weights12);
        self.hidden2 = 1.0/(1.0+exp(-self.beta*self.hidden2))
        self.hidden2 = concatenate((self.hidden2,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden2,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = sum(exp(outputs),axis=1)*ones((1,shape(outputs)[0]))
            return transpose(transpose(exp(outputs))/normalisers)
        else:
            print "error"

