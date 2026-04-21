import matplotlib.pyplot as plt
from matplotlib import colormaps
plt.register_cmap = colormaps.register

import olympus
from olympus.datasets import Dataset

try:
    d = Dataset(kind='suzuki_i')
    print("Suzuki I found")
    print("Param space:", d.param_space)
    for p in d.param_space:
        print(f"Param: {p.name}, Type: {p.type}")
except Exception as e:
    print(f"Suzuki I error: {e}")

try:
    d4 = Dataset(kind='suzuki_iv')
    print("Suzuki IV found")
    print("Param space:", d4.param_space)
    for p in d4.param_space:
        print(f"Param: {p.name}, Type: {p.type}")
except Exception as e:
    print(f"Suzuki IV error: {e}")
