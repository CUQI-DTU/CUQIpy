from cuqi.experimental.mcmc import SamplerNew

class DirectNew(SamplerNew):
    """ Direct sampler

    This sampler is used to sample from a target distribution directly. It simply calls the sample method of the target object to generate a sample.

    Parameters
    ----------
    target : Distribution
        The target distribution to sample from.

    """
            
    def _initialize(self):
        pass

    def validate_target(self):
        try:
            self.target.sample()
        except:
            raise TypeError("Direct sampler requires a target with a sample method.")
        
    def step(self):
        self.current_point = self.target.sample()

    def tune(self, skip_len, update_count):
        pass     
