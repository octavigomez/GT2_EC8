import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=42)

def normals_srl(srl, unc, srl_calc):
    srl_pdf = stats.norm.pdf(srl_calc,srl,unc)
    return srl_pdf

def randoms_srl(calc_points, srl_pdf, num_samples):
    random_srl = rng.choice(calc_points, size=num_samples, p=srl_pdf/np.sum(srl_pdf))
    return random_srl

def normal_m(m, sd_m, m_range):
    m_pdf = stats.norm.pdf(m_range, m, sd_m)
    return m_pdf

def density_to_prob(bins, probabilities):
    dx = bins[1]-bins[0]
    p = dx*probabilities
    p = p/np.sum(p)
    return p