import numpy as np
import pandas as pd

animals = ('dog', 'eagle', 'bat')
methods = ('disease','dive','teeth')
priors = (1/3,1/3,1/3) # (P(dog), P(eagle), P(bat))
dog_probs = (0.15,0.05,0.8) # (P(disease|dog), P(dive|dog), P(teeth|dog))
eagle_probs = (0.1,0.8,0.1) # (P(disease|eagle), P(dive|eagle), P(teeth|eagle))
bat_probs = (0.6,0.2,0.2) # (P(disease|bat), P(dive|bat), P(teeth|bat))
all_probs = (dog_probs,eagle_probs,bat_probs)

analytic_table = pd.DataFrame((np.array(all_probs)*np.array(priors)).T,\
    columns = animals, index = methods)

num_sims = int(1e7)

def draw_prior(animals,priors):
    return np.random.choice(animals, p=priors)

def draw_model(methods, suspect, all_probs, animals):
    return np.random.choice(methods, p=all_probs[animals.index(suspect)])

def draw_joint(methods, animals, all_probs, priors):
    s = draw_prior(animals, priors)
    w = draw_model(methods, s, all_probs, animals)
    return s + "_" + w
def simulator(*args, num_sims = int(1e4)):
    return [draw_joint(*args) for _ in range(num_sims)]

sims = simulator(methods, animals, all_probs, priors, num_sims = num_sims)
outcome,count=np.unique(sims, return_counts=True)
approx_table = pd.DataFrame(columns = animals, index=methods)

for s,c in zip(outcome,count):
    animal = s.split('_')[0]
    method = s.split('_')[1]
    approx_table[animal][method] = c/num_sims
    print(f"Outcome {s} occurred {c} times, with a frequency of {c/num_sims}")
print(f"\nApproximations when N = {num_sims}:")
print(f"{approx_table}\n")
print("Analytic probabilities:")
print(f"{analytic_table}\n")


