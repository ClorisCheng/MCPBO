import numpy as np

attribute_vals = np.loadtxt("attribute_vals.txt")
normalized_attribute_vals = (attribute_vals - attribute_vals.min(0)) / (
    attribute_vals.max(0) - attribute_vals.min(0)
)
np.savetxt("normalized_attribute_vals.txt", normalized_attribute_vals)
