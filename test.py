from plotter import cvt_sum_anim, plot, create_and_save, cvt_animation
from func_collections import *
import time
import cv2
import matplotlib.pyplot as plt

a = (-1, 1, -1, 1)
b = (-10, 10, -10, 10)
c = (-20, 20, -20, 20)
d = (-1.5, 1.5, -1.5, 1.5)
H = (-1, 1, 0, 2)

if __name__ == "__main__":
    im = plot(lambda z: j(z, 1000), H, resolution = (4096, 4096), saturation_cutoff=(1, 1e8))