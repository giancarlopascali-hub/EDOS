import matplotlib.pyplot as plt
from matplotlib import colormaps
import sys

# Robustly patch register_cmap
plt.register_cmap = colormaps.register

import olympus
from olympus.datasets import Dataset
from olympus.datasets import list_datasets

print("Available datasets:", list_datasets())

try:
    d1 = Dataset(kind='suzuki_i')
    print("Suzuki I found")
    print(d1.data.head())
except Exception as e:
    print(f"Suzuki I error: {e}")

try:
    d4 = Dataset(kind='suzuki_iv')
    print("Suzuki IV found")
    print(d4.data.head())
except Exception as e:
    print(f"Suzuki IV error: {e}")
