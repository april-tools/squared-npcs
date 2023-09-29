import io
from typing import Optional

import numpy as np
from PIL import Image as pillow
from matplotlib import pyplot as plt
from tueplots import fonts, figsizes, fontsizes


def setup_tueplots(
        nrows: int,
        ncols: int,
        rel_width: float = 1.0,
        hw_ratio: Optional[float] = None,
        inc_font_size: int = 0,
        **kwargs
):
    font_config = fonts.iclr2023_tex(family='serif')
    if hw_ratio is not None:
        kwargs['height_to_width_ratio'] = hw_ratio
    size = figsizes.iclr2023(rel_width=rel_width, nrows=nrows, ncols=ncols, **kwargs)
    fontsize_config = fontsizes.iclr2023(default_smaller=-inc_font_size)
    rc_params = {**font_config, **size, **fontsize_config}
    plt.rcParams.update(rc_params)
    #plt.rcParams.update({
    #    "axes.prop_cycle": plt.cycler(
    #        color=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    #               "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
    #    ),
    #    "patch.facecolor": "#0173B2"
    #})


def array_to_image(array: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    assert len(array.shape) == 2
    xi, yi = np.mgrid[range(array.shape[0]), range(array.shape[1])]
    setup_tueplots(1, 1, hw_ratio=1.0)
    fig, ax = plt.subplots()
    ax.pcolormesh(xi, yi, array, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    return matplotlib_buffer_to_image(fig)


def matplotlib_buffer_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buffer_to_image(buf)


def buffer_to_image(buf: io.BytesIO) -> np.ndarray:
    with pillow.open(buf, formats=['png']) as fp:
        return np.array(fp, dtype=np.uint8).transpose([2, 0, 1])
