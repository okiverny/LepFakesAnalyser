import matplotlib.pyplot as plt
import numpy as np

# Data from Counting method
counting_data = {
    '1b1p': {'20-30': (2804.2, 87.2), '30-40': (1015.2, 46.3), '>40': (870.7, 35.5)},
    '1b3p': {'20-30': (893.5, 29.1), '30-40': (363.7, 17.4), '>40': (248.8, 13.4)},
    '2b1p': {'20-30': (651.3, 33.0), '30-40': (227.3, 17.5), '>40': (172.1, 12.8)},
    '2b3p': {'20-30': (130.1, 7.6), '30-40': (39.4, 3.6), '>40': (38.6, 3.6)},
}

# Data from Template Fit method
template_fit_data = {
    '1b1p': {'20-30': (2945.7, 0.0), '30-40': (1031.6, 0.0), '>40': (886.1, 0.0)},
    '1b3p': {'20-30': (884.7, 0.0), '30-40': (361.3, 0.0), '>40': (253.6, 0.0)},
    '2b1p': {'20-30': (812.2, 0.0), '30-40': (270.8, 0.0), '>40': (193.9, 0.0)},
    '2b3p': {'20-30': (154.9, 0.0), '30-40': (36.8, 0.0), '>40': (33.4, 0.0)},
}

# Extracting pT bins and categories
pT_bins = ['20-30', '30-40', '>40']
categories = ['1b1p', '1b3p', '2b1p', '2b3p']
colors = ['b', 'g', 'r', 'c']  # Colors for different categories

# Plotting the ratio of Template Fit method over Counting method
fig, ax = plt.subplots()

for i, category in enumerate(categories):
    counting_values = np.array([counting_data[category][pT][0] for pT in pT_bins])
    counting_uncertainties = np.array([counting_data[category][pT][1] for pT in pT_bins])
    
    template_fit_values = np.array([template_fit_data[category][pT][0] for pT in pT_bins])
    
    ratio = template_fit_values / counting_values
    ratio_error = counting_uncertainties / counting_values
    
    # Adding small x-axis shift for visibility
    x_shift = i * 0.02
    
    ax.errorbar(np.arange(len(pT_bins)) + x_shift, ratio, yerr=ratio_error,
                fmt='o', label=category, color=colors[i], capsize=5)

# Setting up the plot
ax.set_xticks(np.arange(len(pT_bins)))
ax.set_xticklabels(pT_bins)
ax.set_xlabel('Tau $p_T$ (GeV)')
ax.set_ylabel('Template Fit / Counting Method')
ax.legend(loc='lower left')

plt.title('Ratio of Njets (Template Fit over Counting Method) in Bins of Tau $p_T$')
plt.grid(True)

plt.savefig('compare.pdf')

