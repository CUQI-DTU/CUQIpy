class DenoiseRegularizer():
    """
    1rst paragraph: What is this class ? What does it represent ?
    2nd paragraph: Connexion between denoisers and proximal operators
    Explain that proximal \subset denoiser. 
    Parameters: 
    """
    
    def __init__(self, denoiser, denoiser_setup = None, strength_reg = 0.1):
        if denoiser_setup is None:
            denoiser_setup = {}
        self.denoiser = denoiser
        self.denoiser_setup = denoiser_setup
        self.strength_reg = strength_reg
        
    def denoise(self, x):
        solution, info = self.denoiser(x, **self.denoiser_setup)
        self.info = info
        return solution 
    
    def gradient(self, x):
        return (x - self.denoise(x))/self.strength_reg
