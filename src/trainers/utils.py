import torch 
import torch.nn as nn
import torch.nn.functional as F
import cProfile
import pstats
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from matplotlib.widgets import Slider
import os

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


def sliderPlot(y, y_hat, batchSize, colormap='turbo'):
    ts = 0
    batch_idx = 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.3)  
    im0 = ax[0].imshow(y[batch_idx, :, :, ts].cpu(), cmap=colormap)
    im1 = ax[1].imshow(y_hat[batch_idx, :, :, ts].cpu().detach(), cmap=colormap)
    im2 = ax[2].imshow(abs(y[batch_idx, :, :, ts].cpu() - y_hat[batch_idx, :, :, ts].cpu().detach()))

    ax[0].set_title("Ground Truth")
    ax[1].set_title("Prediction")
    ax[2].set_title("Absolute Difference")

    ax_slider_ts = plt.axes([0.25, 0.15, 0.5, 0.03]) 
    slider_ts = Slider(ax_slider_ts, 'Timestep', 0, 9, valinit=ts, valstep=1)

    ax_slider_batch = plt.axes([0.25, 0.05, 0.5, 0.03])  
    slider_batch = Slider(ax_slider_batch, 'Batch Index', 0, y.shape[0] - 1, valinit=batch_idx, valstep=1)

    def update(val):
        ts = int(slider_ts.val)
        batch_idx = int(slider_batch.val) 
        im0.set_data(y[batch_idx, :, :, ts].cpu())
        im1.set_data(y_hat[batch_idx, :, :, ts].cpu().detach())
        im2.set_data(abs(y[batch_idx, :, :, ts].cpu() - y_hat[batch_idx, :, :, ts].cpu().detach()))
        fig.canvas.draw_idle()

    slider_ts.on_changed(update)
    slider_batch.on_changed(update)
    #plt.tight_layout()
    plt.show()    

def rollout_temp(model, input, device, tw, steps=180):
    model.eval()
    stacked_pred = input[0, :tw, :, :]  
    with torch.no_grad():
        while stacked_pred.shape[0] < steps:
            output = model(input)
            # stack the output on stacked_pred but take only every first 5 frames
            stacked_pred = torch.cat((stacked_pred, output[0, :tw, :, :]), 0)
            #stacked_pred = torch.cat((stacked_pred, output), 0)
            #print(stacked_pred.shape)   
            input = output
    return stacked_pred


def create_gif2(stacked_true, stacked_pred, output_path, timesteps='all', vertical=False):
    if timesteps == 'all':
        timesteps = stacked_pred.shape[0]
    else:
        timesteps = int(timesteps)
    imgtrue_stacked = torch.flip(stacked_true, dims=[1])
    imgpred_stacked = torch.flip(stacked_pred, dims=[1])
    if vertical:
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    def update_frame(i):
        ax[0].clear()
        ax[1].clear()

        imgtrue = imgtrue_stacked[i, :, :]
        imgpred = imgpred_stacked[i, :, :]

        ax[0].imshow(imgtrue, cmap='RdBu_r', vmin=-1, vmax=1)
        ax[0].set_title("True")
        ax[0].axis('off')

        ax[1].imshow(imgpred, cmap='RdBu_r', vmin=-1, vmax=1)
        ax[1].set_title("Prediction")
        ax[1].axis('off')

        fig.suptitle(f"Step {i + 1}/{timesteps}")

    ani = animation.FuncAnimation(fig, update_frame, frames=timesteps, interval=1)
    if output_path is not None:
        ani.save(output_path, writer='ffmpeg', fps=30)
    plt.close()
    return ani
    
def animate_rollout(stacked_pred, stacked_true, dataset_name, output_path="output/rollout.gif"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    timesteps, _, x_dim, y_dim = stacked_pred.shape
    stacked_pred, stacked_true = stacked_pred.squeeze(1).cpu().numpy(), stacked_true.squeeze(1).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Pred", "True", "Diff"]
    imgs = []
    vmin, vmax = min(stacked_pred.min(), stacked_true.min()), max(stacked_pred.max(), stacked_true.max())

    for ax, title in zip(axes, titles):
        #print(ax, title)
        img = ax.imshow(np.zeros((x_dim, y_dim)), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        imgs.append(img)

    axes[0].set_ylabel(r'$\sqrt{v_x^2 + v_y^2}$')
    imgs[2].set_cmap('magma')
    
    def init():
        imgs[0].set_data(stacked_pred[0])
        imgs[1].set_data(stacked_true[0])
        #imgs[2].set_data(torch.zeros([stacked_true.shape[0], stacked_true.shape[1]]))
        return imgs

    def update(frame):
        imgs[0].set_data(stacked_pred[frame])
        imgs[1].set_data(stacked_true[frame])
        imgs[2].set_data(np.abs(stacked_true[frame] - stacked_pred[frame]))

        fig.suptitle(f"Dataset: {dataset_name}, timestep {frame + 1}")
        #fig.suptitle(f"Dataset: -, timestep {frame + 1}")
        return imgs

    ani = animation.FuncAnimation(fig, update, frames=timesteps, init_func=init, blit=False, interval=50)
    #print(output_path)
    ani.save(output_path, writer="ffmpeg")
    plt.close()

def magnitude_vel(x):
    magnitude = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    return magnitude.unsqueeze(1)

def rollout(front, model, length):
    model.eval()
    preds = []
    with torch.no_grad():
        pred = model(front)
        preds.append(pred)
        for i in range(length - 1):
            pred = model(pred)
            preds.append(pred)
    preds = torch.cat(preds, dim=0)
    return preds
