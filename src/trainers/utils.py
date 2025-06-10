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
    if timesteps > stacked_true.shape[0]:
        timesteps = stacked_true.shape[0]
    stacked_pred, stacked_true = stacked_pred.squeeze(1).cpu().numpy(), stacked_true.squeeze(1).cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Pred", "True", "Diff"]
    imgs = []
    vmin, vmax = min(stacked_pred.min(), stacked_true.min()), max(stacked_pred.max(), stacked_true.max())

    for ax, title in zip(axes, titles):
        #print(ax, title)
        img = ax.imshow(np.zeros((x_dim, y_dim)), cmap='viridis', vmin=0, vmax=vmax)#, vmin=vmin, vmax=vmax)
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
        return imgs

    def update(frame):
        #vmin_frame = min(stacked_pred[frame].min(), stacked_true[frame].min())
        vmax_frame = max(stacked_pred[frame].max(), stacked_true[frame].max())

        for img in imgs:
            #img.set_clim(vmin_frame, vmax_frame)
            img.set_clim(vmin=0, vmax=vmax_frame)

        imgs[0].set_data(stacked_pred[frame])
        imgs[1].set_data(stacked_true[frame])
        imgs[2].set_data(np.abs(stacked_true[frame] - stacked_pred[frame]))

        fig.suptitle(f"Dataset: {dataset_name}, timestep {frame + 1}\n"
                     #f"vmin: {vmin_frame:.4f}, vmax: {vmax_frame:.4f}")
                        f"vmax: {vmax_frame:.3f}")
        return imgs

    ani = animation.FuncAnimation(fig, update, frames=timesteps, init_func=init, blit=False, interval=75)
    #print(output_path)
    ani.save(output_path, writer="ffmpeg")
    plt.close()

def magnitude_vel(x):
    magnitude = torch.sqrt(x[:, :, 0]**2 + x[:, :, 1]**2)
    return magnitude.unsqueeze(2)

def rollout_det(front, model, steps):
    model.eval()
    preds = []
    preds.append(front)

    with torch.no_grad():
        pred = model(front)
        preds.append(pred)
        for i in range(steps - 2):
            pred = model(pred)
            preds.append(pred)

    preds = torch.cat(preds, dim=1)
    return preds

def rollout_prb(front, model, steps, perturb_func, perturbation_strength, int_steps):
    model.eval()
    preds = []
    preds.append(front)
    with torch.no_grad():
        xt = perturb_func(front, perturbation_strength=perturbation_strength)
        for _ in range(steps - 1):
            for i, t in enumerate(torch.linspace(0, 1, int_steps), start=1):
                pred = model(xt, t.to(xt.device).expand(xt.size(0)))
                xt = xt + (1 / steps) * pred
            preds.append(xt)
    preds = torch.cat(preds, dim=1)
    return preds

def compute_energy_enstrophy_spectra(u, v, dataset_name="", Lx=1.0, Ly=1.0):

    assert u.shape == v.shape, "u and v must have the same shape"
    device = u.device
    nx, ny = u.shape
    dx = Lx / nx
    dy = Ly / ny

    if dataset_name == "pdebench-incomp": #only this dataset has other than periodic BC
        # so we use hanning window
        window_x = torch.hann_window(nx, periodic=False, device=device)
        window_y = torch.hann_window(ny, periodic=False, device=device)
        window = torch.outer(window_x, window_y)
        u = u * window
        v = v * window

    # Wavenumbers
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K = torch.sqrt(K2)

    u_hat = torch.fft.fft2(u)
    v_hat = torch.fft.fft2(v)

    # Energy and enstrophy spectra
    E_k = 0.5 * (torch.abs(u_hat)**2 + torch.abs(v_hat)**2)
    w_hat = 1j * KX * v_hat - 1j * KY * u_hat
    Z_k = 0.5 * torch.abs(w_hat)**2

    K_flat = K.flatten()
    E_flat = E_k.flatten()
    Z_flat = Z_k.flatten()

    k_max = K_flat.max()
    n_bins = nx // 2
    k_bins = torch.linspace(0, k_max, n_bins + 1, device=device)
    k_bins_center = 0.5 * (k_bins[:-1] + k_bins[1:])

    E_spectrum = torch.zeros(n_bins, device=device)
    Z_spectrum = torch.zeros(n_bins, device=device)
    counts = torch.zeros(n_bins, device=device)

    # Bin points
    bin_idx = torch.bucketize(K_flat, k_bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid]
    E_flat = E_flat[valid]
    Z_flat = Z_flat[valid]

    for i in range(n_bins):
        mask = bin_idx == i
        E_spectrum[i] = E_flat[mask].sum()
        Z_spectrum[i] = Z_flat[mask].sum()
        counts[i] = mask.sum()

    counts[counts == 0] = 1  # Avoid division by zero
    E_spectrum /= counts
    Z_spectrum /= counts

    return k_bins_center, E_spectrum, Z_spectrum