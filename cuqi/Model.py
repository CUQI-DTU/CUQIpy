class Generic(object):
    
    def __init__(self,forward,dim=[]):
        
        #Check if input is callable
        if callable(forward) is not True:
            raise TypeError("Forward needs to be callable function of some kind")
            
        #Store forward func
        self.forward_func = forward
        
        #Store dimension
        self.dim = dim
               
    def forward(self, x):
        return self.forward_func(x)
    
class Linear(Generic):
    # Linear forward model with forward and adjoint.
    # Only accepts functions for now
    
    def __init__(self,forward,adjoint,dim=[]):
        
        #Store exactly as with generic
        Generic.__init__(self,forward,dim)
        
        #Check if input is callable
        if callable(adjoint) is not True:
            raise TypeError("Adjoint needs to be callable function of some kind")
            
        #Add adjoint
        self.adjoint = adjoint
        
        
