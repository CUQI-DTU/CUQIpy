class Denoiser():
    
    def __init__(self, denoiser, denoiser_setup, strength_reg):
        self.denoiser = denoiser
        self.denoiser_setup = denoiser_setup
        self.strength_reg = strength_reg
        
    def denoise(self, x):
        solution, info = self.denoiser(x, **self.denoiser_setup)
        self.info = info
        return solution 
    
    def grad_reg(self, x):
        return (x - self.denoise(x))/self.strength_reg