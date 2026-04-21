import matplotlib.pyplot as plt
from matplotlib import colormaps
plt.register_cmap = colormaps.register

from olympus.datasets import Dataset
try:
    d = Dataset(kind='suzuki_i')
    print("Dataset suzuki_i loaded")
except Exception as e:
    print(f"Error loading suzuki_i: {e}")
