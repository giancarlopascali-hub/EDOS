import matplotlib
print("Matplotlib version:", matplotlib.__version__)
import matplotlib.pyplot as plt
try:
    import matplotlib.cm as cm
    print("cm has register_cmap:", hasattr(cm, 'register_cmap'))
except: pass

try:
    from matplotlib import colormaps
    print("colormaps has register:", hasattr(colormaps, 'register'))
except: pass
