import torch 
import torch.nn as nn
import torch.nn.functional as F
import cProfile
import pstats
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def get_meshgrid(shape, tstep):
    batch_size, x_size, y_size, t_size, _ = shape
    #print(t_size)
    #batch_size, x_size, y_size, t_size = shape[0], shape[1], shape[2], shape[3]
    x_grid = torch.linspace(0, 1, x_size)
    x_grid = x_grid.reshape(1, x_size, 1, 1, 1).repeat([batch_size, 1, y_size, t_size, 1])

    y_grid = torch.linspace(0, 1, y_size)
    y_grid = y_grid.reshape(1, 1, y_size, 1, 1).repeat([batch_size, x_size, 1, t_size, 1])

    t_grid = torch.linspace(0, 1, t_size) + tstep
    #print(t_grid)
    t_grid = t_grid.reshape(1, 1, 1, t_size, 1).repeat([batch_size, x_size, y_size, 1, 1])

    grid = torch.concat((x_grid, y_grid, t_grid), dim=-1).float()
    return grid

def numworkersTest(trainingdata, workerslist, bs):
    for gpu_on in [True, False]:
        for workers in workerslist:
            profiler = cProfile.Profile()
            profiler.enable()
            trainingDataLoader = DataLoader(
                trainingdata, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=gpu_on
            )
            print(f"GPU: {gpu_on}, Workers: {workers}", end=' - ')
            for batch_idx, (data, target) in enumerate(trainingDataLoader):
                break
            profiler.disable()
            profiler_stats = pstats.Stats(profiler)
            total_time = sum(stat[4] for stat in profiler.getstats()) 
            print(f"Tottime: {total_time:.5f} s")

def createGif(data, filename, colormap='viridis'):
    fig = plt.figure(figsize=(5,5))    
    cmin= torch.min(data)
    cmax = torch.max(data)
    im = plt.imshow(data[:, :, 0], 
                    animated=True,
                    cmap=colormap,  
                    vmin=cmin, 
                    vmax=cmax)

    plt.tight_layout()
    plt.axis('off')

    def init():
        im.set_data(data[:, :, 0])
        return im,

    def animate(i):
        im.set_array(data[:, :, i])
        return im,

    anim = animation.FuncAnimation(fig,
                                    animate,
                                    init_func=init,
                                    frames=np.shape(data)[2], 
                                    interval=100, 
                                    blit=True)
    outputFolder = 'output/'
    anim.save(outputFolder + filename + ".gif")  
    plt.close(fig)