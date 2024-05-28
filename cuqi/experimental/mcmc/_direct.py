from cuqi.experimental.mcmc import SamplerNew

class Direct(SamplerNew):
            
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
