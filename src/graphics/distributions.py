from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt

from pcs.models import PC
from pcs.utils import retrieve_default_dtype

from graphics.utils import setup_tueplots, matplotlib_buffer_to_image


def plot_bivariate_samples_hmap(
        data: np.ndarray,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        zm: float = 0.0,
        nbins: int = 256
) -> np.ndarray:
    setup_tueplots(1, 1, hw_ratio=1.0)
    if xlim is None:
        xlim = data[:, 0].min(), data[:, 0].max()
    if ylim is None:
        ylim = data[:, 1].min(), data[:, 1].max()
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    fig, ax = plt.subplots()
    xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins * 1j, ylim[0]:ylim[1]:nbins * 1j]
    zi, _, _ = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=nbins,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        density=True)
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', vmin=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return matplotlib_buffer_to_image(fig)


def plot_bivariate_discrete_samples_hmap(
        data: np.ndarray,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if xlim is None:
        xlim = np.min(data[:, 0]), np.max(data[:, 0])
    if ylim is None:
        ylim = np.min(data[:, 0]), np.max(data[:, 1])
    setup_tueplots(1, 1, hw_ratio=1.0)
    fig, ax = plt.subplots()
    zi, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=[xlim[1] - xlim[0] + 1, ylim[1] - ylim[0] + 1],
        range=[[xlim[0], xlim[1] + 1], [ylim[0], ylim[1] + 1]],
        density=True)
    yi, xi = np.meshgrid(xedges[:-1], yedges[:-1])
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', vmin=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return matplotlib_buffer_to_image(fig)


def discrete_samples_hmap(
        data: np.ndarray,
        xlim: Optional[Tuple[int, int]] = None,
        ylim: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    if xlim is None:
        xlim = np.min(data[:, 0]), np.max(data[:, 0])
    if ylim is None:
        ylim = np.min(data[:, 0]), np.max(data[:, 1])
    zi, xedges, yedges = np.histogram2d(
        data[:, 0], data[:, 1],
        bins=[xlim[1] - xlim[0] + 1, ylim[1] - ylim[0] + 1],
        range=[[xlim[0], xlim[1] + 1], [ylim[0], ylim[1] + 1]],
        density=True)
    yi, xi = np.meshgrid(xedges[:-1], yedges[:-1])
    return zi.reshape(xi.shape)


def kde_samples_hmap(
        data: np.ndarray,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        zm: float = 0.0,
        nbins: int = 256,
        *,
        bandwidth: float = 0.2
) -> np.ndarray:
    if xlim is None:
        xlim = data[:, 0].min(), data[:, 0].max()
    if ylim is None:
        ylim = data[:, 1].min(), data[:, 1].max()
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins * 1j, ylim[0]:ylim[1]:nbins * 1j]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1)\
        .astype(retrieve_default_dtype(numpy=True), copy=False)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(data)
    zi = np.exp(kde.score_samples(xy)).reshape(xi.shape)
    return zi


@torch.no_grad()
def bivariate_pdf_heatmap(
        model: PC,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        zm: float = 0.0,
        nbins: int = 256,
        batch_size: Optional[int] = None,
        variables: Optional[Tuple[int, int]] = None,
        device: Optional[Union[str, torch.device]] = None,
) -> np.ndarray:
    if device is None:
        device = 'cpu'
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    xi, yi = np.mgrid[xlim[0]:xlim[1]:nbins * 1j, ylim[0]:ylim[1]:nbins * 1j]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1) \
        .astype(retrieve_default_dtype(numpy=True), copy=False)
    if model.num_variables < 1:
        raise ValueError("A PC defined over two variables is required")
    elif model.num_variables == 2:
        xy = torch.from_numpy(xy).to(device)
        if isinstance(model, PC):
            zi = model.log_prob(xy)
        else:
            zi = model().log_prob(xy)
    else:
        if batch_size is None:
            batch_size = 128
        if variables is None or len(variables) != 2:
            raise ValueError("The two variables to show the PDF of cannot be None")
        # Perform variable marginalization
        points = TensorDataset(xy[0], xy[1])
        points = DataLoader(points, batch_size=batch_size, shuffle=False, drop_last=False)
        mar_mask = torch.ones(1, model.num_variables, dtype=torch.bool, device=device)
        mar_mask[:, variables] = False
        zi = list()
        for xf, yf in points:
            samples = torch.zeros(size=(xf.shape[0], model.num_variables), dtype=xf.dtype)
            samples[:, variables[0]] = xf
            samples[:, variables[1]] = yf
            samples.to(device)
            log_probs = model.log_marginal_prob(samples, mar_mask)
            zi.append(log_probs)
        zi = torch.concatenate(zi, dim=0)
    zi = torch.exp(zi).cpu().numpy().reshape(xi.shape)
    return zi


@torch.no_grad()
def bivariate_pmf_heatmap(
        model: PC,
        xlim: Tuple[int, int],
        ylim: Tuple[int, int],
        device: Optional[Union[str, torch.device]] = None
) -> np.ndarray:
    xi, yi = np.mgrid[xlim[0]:xlim[1] + 1, ylim[0]:ylim[1] + 1]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1)
    if device is None:
        device = 'cpu'
    xy = torch.from_numpy(xy).to(device)
    zi = model.log_prob(xy)
    zi = torch.exp(zi).cpu().numpy().reshape(xi.shape)
    return zi
