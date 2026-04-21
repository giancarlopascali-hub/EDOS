import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Patch matplotlib for olympus compatibility
if not hasattr(plt, 'register_cmap'):
    plt.register_cmap = cm.register_cmap

import olympus
from olympus.datasets import Dataset
from olympus.datasets import list_datasets

print("Available datasets:", list_datasets())

try:
    d1 = Dataset(kind='suzuki_i')
    print("Suzuki I found")
    print(d1.data.head())
    print("Param space:", d1.param_space)
except Exception as e:
    print(f"Suzuki I error: {e}")

try:
    d4 = Dataset(kind='suzuki_iv')
    print("Suzuki IV found")
    print(d4.data.head())
    print("Param space:", d4.param_space)
except Exception as e:
    print(f"Suzuki IV error: {e}")
