import json 
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def load_dataset(json_path):

    with open(json_path, 'r') as f:
        return json.load(f)
    

def normalize_answer(s):
    """
    Lowercase and remove punctuation (except for negative signs), articles,
    and extra whitespace. This normalization is similar to TriviaQA/SQuAD,
    but preserves negatives for the SVAMP dataset.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text):
        # Exclude the minus sign from punctuation removal.
        exclude = set(string.punctuation) - {'-'}
        return ''.join(ch for ch in text if ch not in exclude)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))

#Estimator for missing mass 
def good_turing(n_r_counts):
    filtered_counts = {r: n for r, n in n_r_counts.items() if n > 0}
    N = sum(r * n for r, n in filtered_counts.items())
    N1 = filtered_counts.get(1, 0)

    if N == 0:
        return 1.0  # Max uncertainty if no data
    return min(max(N1 / N, 0.0), 1.0)

#Estimator for missing mass derivative 
def estimate_derivative_missing_mass(freq_counts: dict) -> float:
    """
    Estimate dP_EE/dN ≈ -2 * N2 / N^2,
    where N2 = freq_counts.get(2,0), N = total samples.
    """
    N = sum(r * nr for r, nr in freq_counts.items())
    N2 = freq_counts.get(2, 0)
    return (2 * N2 / (N ** 2)) if N > 0 else 0.0


# SMoothed good turing with fallbakc mechanism 
def simple_good_turing(n_r_counts):

    filtered_counts = {r: n for r, n in n_r_counts.items() if n > 0}
    N = sum(r * n for r, n in filtered_counts.items())

    if N == 0:
        return 1.0  # No samples, maximal uncertainty

    r_vals = np.array(sorted(filtered_counts.keys()))
    n_r = np.array([filtered_counts[r] for r in r_vals])

    if len(r_vals) >= 2:
        # Safe regression with at least 2 distinct frequency points
        log_r = np.log(r_vals)
        log_n_r = np.log(n_r)
        slope, intercept = np.polyfit(log_r, log_n_r, 1)
        n1_smooth = np.exp(intercept)  # extrapolation at r=1
    else:
        # Principled fallback: single frequency case
        r_single = r_vals[0]
        n_single = n_r[0]
        if r_single == 1:
            n1_smooth = n_single  # observed directly
        else:
            # conservative fallback to small unseen probability
            n1_smooth = np.random.uniform(0, 0.05)  # small regularized estimate

    P0 = n1_smooth / N
    return min(max(P0, 0.0), 1.0)
