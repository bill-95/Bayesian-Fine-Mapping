import pandas as pd
import numpy as np
from itertools import combinations
import scipy
from scipy.stats import multivariate_normal as mn
import matplotlib.pyplot as plt

def load_data():
    z_scores = pd.read_csv('data/zscore.csv', index_col=0)
    snp_idx = list(z_scores.index)
    LD = pd.read_csv('data/LD.csv', index_col=0)
    LD = LD.loc[z_scores.index,:] # match index
    z_scores_np = z_scores.values.reshape(-1)
    LD_np = LD.values
    M = len(z_scores_np)
    return snp_idx, z_scores_np, LD_np, M

# 1. load in data
snp_idx, z_scores_np, LD_np, M = load_data()
N = 498 # sample size
s2_lambda = 0.005 # user-defined prior variance

# 2. calculate all sigma_cc terms
sigma_cc_3 = N*s2_lambda*np.identity(3)
sigma_cc_2 = N*s2_lambda*np.identity(2)
sigma_cc_1 = N*s2_lambda
all_sigma_ccs = {3: sigma_cc_3, 2: sigma_cc_2, 1: sigma_cc_1}

# 3. calculate all configurations
configs_3_snp = list(map(list, combinations(range(M), 3)))
configs_2_snp = list(map(list, combinations(range(M), 2)))
configs_1_snp = list(map(list, combinations(range(M), 1)))

# 4. calculate all priors
prior_3 = (1/M)**3 * ((M-1)/M)**(M-3) # prior of each configuration with 3 SNPs
prior_2 = (1/M)**2 * ((M-1)/M)**(M-2) # prior of each configuration with 2 SNPs
prior_1 = (1/M) * ((M-1)/M)**(M-1) # prior of each configuration with 1 SNPs
all_priors = {3: prior_3, 2: prior_2, 1: prior_1}

# 5. functions for calculating bayes factors
def get_config_bf(config, k):
    LD_cc = LD_np[config][:, config]
    cov_cc = LD_cc + LD_cc.dot(all_sigma_ccs[k]).dot(LD_cc)
    z_cc = z_scores_np[config]
    try:
        llh_c = mn.pdf(z_cc, mean=[0]*k, cov=cov_cc)
        llh_nc = mn.pdf(z_cc, mean=[0]*k, cov=LD_cc)
    # return None for ones which have determinant of 0
    except np.linalg.LinAlgError as err:
        if 'singular matrix' in str(err):
            return None
    llh_ratio = llh_c/llh_nc
    return llh_ratio

# 6. get the bayes factors for each configurations using above function
bf_3_snp = []
for config in configs_3_snp:
    config_bf = get_config_bf(config, 3)
    bf_3_snp.append(config_bf)

bf_2_snp = []
for config in configs_2_snp:
    config_bf = get_config_bf(config, 2)
    bf_2_snp.append(config_bf)

bf_1_snp = []
for config in configs_1_snp:
    config_bf = get_config_bf(config, 1)
    bf_1_snp.append(config_bf)

# 7. filter bayes factors and configs passing singular matrix check
bf_3_snp_filt = [bf_3_snp[i] for i in range(len(bf_3_snp)) if bf_3_snp[i] != None]
bf_2_snp_filt = [bf_2_snp[i] for i in range(len(bf_2_snp)) if bf_2_snp[i] != None]
bf_1_snp_filt = bf_1_snp # no need to filter this
configs_3_snp_filt = [configs_3_snp[i] for i in range(len(bf_3_snp)) if bf_3_snp[i] != None]
configs_2_snp_filt = [configs_2_snp[i] for i in range(len(bf_2_snp)) if bf_2_snp[i] != None]
configs_1_snp_filt = configs_1_snp # no need to filter this

# 8. get marginal of filtered bayes factors
marginal_llh = sum(bf_3_snp_filt + bf_2_snp_filt + bf_1_snp_filt)

# 9. function for calculating posterior probabilities
def get_posterior(bf, k):
    return bf * all_priors[k] / marginal_llh

# 10. get the posterior for each configurations using above function
posterior_3_snp = []
for bf in bf_3_snp_filt:
    p = get_posterior(bf, 3)
    posterior_3_snp.append(p)

posterior_2_snp = []
for bf in bf_2_snp_filt:
    p = get_posterior(bf, 2)
    posterior_2_snp.append(p)

posterior_1_snp = []
for bf in bf_1_snp_filt:
    p = get_posterior(bf, 1)
    posterior_1_snp.append(p)
all_posteriors = posterior_3_snp + posterior_2_snp + posterior_1_snp

# 11. scale posteriors and visualization
all_posteriors_scaled = [ele*1000000 for ele in all_posteriors]
plt.figure(num=None, figsize=(10, 5), dpi=100, facecolor='w', edgecolor='k')
plt.scatter(range(len(all_posteriors_scaled)), sorted(all_posteriors_scaled))
plt.xlabel('Sorted configurations')
plt.ylabel('Configuration posterior')
plt.show()

# 12. calculate PIP for each SNP
# create a mapping between a SNP's index and all configs containing the SNP
snp_to_config_idx = {k:[] for k in range(M)}
all_configs_filt = configs_3_snp_filt + configs_2_snp_filt + configs_1_snp_filt
for i in range(len(all_configs_filt)):
    for snp in all_configs_filt[i]:
        snp_to_config_idx[snp].append(i)
# convert posteriors to numpy array for fast indexing for calculating PIP
all_posteriors_np = np.array(all_posteriors)
# get the denominator for PIP
all_posteriors_sum = all_posteriors_np.sum()
# calculate PIP for each SNP
all_pips = []
for i in range(M):
    pip = all_posteriors_np[snp_to_config_idx[i]].sum() / all_posteriors_sum
    all_pips.append(pip)

# 13. visualizing the PIP and -log10 p-values
p_values = scipy.stats.norm.sf(abs(z_scores_np))
log_p_val = -np.log10(p_values)
colors = []
for ele in snp_idx:
    if ele in {'rs10104559', 'rs1365732', 'rs12676370'}: # causal SNP set
        colors.append('tab:red')
    else:
        colors.append('tab:blue')
fig, axs = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(10)
axs[0].set_title('-log10p and PIP')
axs[0].scatter(range(M), log_p_val, c=colors)
axs[0].set_ylabel('-log10p')
axs[1].scatter(range(M), all_pips, c=colors)
axs[1].set_ylabel('PIP')
axs[1].set_xlabel('SNP')
fig.tight_layout()
fig.show()