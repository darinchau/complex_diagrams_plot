from plotter import plot, create_and_save
from func_collections import *
import time

a = (-1, 1, -1, 1)
b = (-10, 10, -10, 10)
c = (-20, 20, -20, 20)

if __name__ == "__main__":
    t = time.time()
    for i in range(9):
        mag = 20 - 0.1 * i if i < 190 else 1 / ( 1 + 0.1 * i) 
        create_and_save(lambda z: Newtons_Attractor(z, lambda z: z ** 3 - 1, lambda z: 3 * z ** 2, 150), (-mag, mag, -mag, mag), save_as = "./z3-1 results/z3res{}.png".format(i))
        if i % 10 == 0: print("Created {}".format(i))
    print(time.time() - t)