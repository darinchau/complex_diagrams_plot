import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

__sat_cut = 1

# Returns the principle argument of z (actually takes and returns a numpy array)
def complex_to_HSV(z: np.ndarray):
    # Use broadcasting magic to convert the grid into the right shape
    res = np.zeros(z.shape + (3,))

    # 0 = H, 1 = S, 2 = V
    # Change to degree and normalize to [0, 179]
    res[:, : ,0] = np.angle(z, deg = True) + 180
    res[:, : ,0] *= 180 / 360

    global __sat_cut
    mag = np.abs(z)
    res[:, :, 1][mag != 0] = __sat_cut / mag[mag != 0] * 255
    res[:, :, 1][mag < __sat_cut] = 255
    
    res[:, :, 2]  = mag / __sat_cut * 255
    res[:, :, 2][mag >= __sat_cut] = 255

    result = np.array(res, dtype = np.uint8)
    return result

def create_img(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), saturation_cutoff = 1):
    # Readability
    x_min, x_max, y_min, y_max = plot_range

    # Create the plot of f(x) = x first
    x_plot = x_min + (x_max - x_min) / resolution[1] * np.arange(resolution[1])
    y_plot = (y_min + (y_max - y_min) / resolution[0] * np.arange(resolution[0])) * 1j
    y_plot = y_plot.reshape(-1, 1)
    z = x_plot + y_plot

    # # Let numpy do it's thing
    t1 = time.time()
    result = f(z)
    time_taken = time.time() - t1

    # # Convert complex number to HSV
    global __sat_cut
    __sat_cut = saturation_cutoff
    res = complex_to_HSV(result)

    # Show graphics using opencv and matplotlib
    image = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return image, time_taken

# Plot range: (x_min, x_max, y_min, y_max)
# Saturation cutoff affects the color scheme of the plot, specifically the darkness of numbers around 0 and the brightness of the colors around infinity
def plot(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), saturation_cutoff = 1):
    image, time_taken = create_img(f, plot_range, resolution, saturation_cutoff)
    plt.figure()
    plt.imshow(image)
    plt.show()
    return time_taken

# Just saves the image
def create_and_save(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), save_as = None, saturation_cutoff = 1):
    image, time_taken = create_img(f, plot_range, resolution, saturation_cutoff)
    plt.imsave(save_as, image)
    return time_taken