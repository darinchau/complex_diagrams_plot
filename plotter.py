import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

__sat_cut = (1, 1)

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
    res[:, :, 1][mag != 0] = __sat_cut[1] / mag[mag != 0] * 255
    res[:, :, 1][mag < __sat_cut[1]] = 255
    
    res[:, :, 2]  = mag / __sat_cut[0] * 255
    res[:, :, 2][mag >= __sat_cut[0]] = 255

    result = np.array(res, dtype = np.uint8)
    return result

def create_img(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), saturation_cutoff = 1):
    # Readability
    x_min, x_max, y_min, y_max = plot_range

    # Create the plot of f(x) = x first
    x_plot = x_min + (x_max - x_min) / resolution[1] * np.arange(resolution[1])
    y_plot = (y_max + (y_min - y_max) / resolution[0] * np.arange(resolution[0]))
    y_plot = y_plot.reshape(-1, 1) * 1j
    z = x_plot + y_plot

    # # Let numpy do it's thing
    result = f(z)

    # # Convert complex number to HSV
    global __sat_cut
    __sat_cut = saturation_cutoff
    res = complex_to_HSV(result)

    # Show graphics using opencv and matplotlib
    image = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return image


# Plot range: (x_min, x_max, y_min, y_max)
# Saturation cutoff affects the color scheme of the plot, specifically the darkness of numbers around 0 and the brightness of the colors around infinity
def plot(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), saturation_cutoff = (1,1e5)):
    image = create_img(f, plot_range, resolution, saturation_cutoff)
    plt.figure()
    plt.imshow(image)
    plt.show()
    return image

# Just saves the image
def create_and_save(f, plot_range: tuple = (-1, 1, -1, 1), resolution: tuple = (1080, 1080), save_as = None, saturation_cutoff = (1, 1e5)):
    image = create_img(f, plot_range, resolution, saturation_cutoff)
    plt.imsave(save_as, image)
    return image

# f is a function that takes in z and the iteration variable i
# plot range n is another function that evaluates to the plot range given the iteration variable i
def cvt_animation(f, plot_range_n, i_range: range, fps = 60, save_as = 'project.avi', saturation_cutoff = (1, 1e5), print_every = 1):
    img_array = []
    lpl = 0
    for i in i_range:
        img = create_img(lambda z: f(z, i), plot_range_n(i), saturation_cutoff = saturation_cutoff)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

        # Print a notification thing every x frames
        if print_every > 0 and i % print_every == 0:
            p_st = "Created frame {}".format(i)
            print("\b" * lpl + p_st, flush = True, end = "")
            lpl = len(p_st)
    
    out = cv2.VideoWriter(save_as,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return

# Optimized implementation for implementing videos of power serieses or things that are based on iterations not zoom range
def cvt_sum_anim(summand, plot_range, i_range: range, fps = 60, save_as = 'project.avi', saturation_cutoff = (1, 1e5), print_every = 1, resolution: tuple = (1080, 1080)):
    # Image array and last print length
    img_array = []
    lpl = 0
    x_min, x_max, y_min, y_max = plot_range

    # Create the plot of f(x) = x first
    x_plot = x_min + (x_max - x_min) / resolution[1] * np.arange(resolution[1])
    y_plot = (y_min + (y_max - y_min) / resolution[0] * np.arange(resolution[0])) * 1j
    y_plot = y_plot.reshape(-1, 1)
    z = x_plot + y_plot

    # Initiate result
    res = np.zeros_like(z)

    # Set saturation cutoff
    global __sat_cut
    __sat_cut = saturation_cutoff

    for i in i_range:
        res += summand(z, i)
        im = complex_to_HSV(res)
        img = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

        # Print a notification thing every x frames
        if print_every > 0 and i % print_every == 0:
            p_st = "Created frame {}".format(i)
            print("\b" * lpl + p_st, flush = True, end = "")
            lpl = len(p_st)
    
    out = cv2.VideoWriter(save_as,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return    