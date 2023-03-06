import random
class Bandit:

    def __init__(self, mu, sigma):

        self.mu = mu                        # mean of rv's
        self.sigma = sigma                  # sdev of rv's
        self.n = 0                          # number of rv's generated, incr
        self.xn = 0                         # rv from generator
        self.mean = 0                       # sample mean
        self.variance = 0                   # sample variance
        self.sample_variance = 0            # sample sample variance -
        self.existing_aggregate = [0,0,0]   # storage for Welford's algorithm
    
    def play(self):                         # method returns result from play
        self.n += 1
        self.xn = random.normalvariate(self.mu,self.sigma)
        self.existing_aggregate = self.update(self.existing_aggregate, self.xn)
        return self.xn
    
    def get_statistics(self):
        return self.finalize(self.existing_aggregate)
        
    # From https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self,existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1 
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        return (count, mean, M2)

    # retrieve the mean, variance and sample variance from an aggregate
    def finalize(self,existingAggregate):
        (count, mean, M2) = existingAggregate
        (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
        if count < 2:
            return float('nan')
        else:
            return (mean, variance, sampleVariance)

            
        