"""Plot spatial kernels, numerical frequency responses and step responses.

Install the optional plotting dependency with ``pip install -e '.[plots]'``.
All kernel and response values are calculated by this project; Matplotlib is
used only to draw the resulting curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from interpolation_study import kernels


def main() -> None:
    x = np.linspace(-4.5, 4.5, 9001)
    dx = x[1] - x[0]
    definitions = {
        "Nearest box": (lambda z: kernels.nearest(z), 0.5),
        "Bilinear": (kernels.linear, 1.0),
        "Keys bicubic (a=-0.5)": (kernels.keys_cubic, 2.0),
        "Mitchell (B=C=1/3)": (kernels.mitchell_netravali, 2.0),
        "Cubic B-spline basis": (kernels.cubic_bspline, 2.0),
        "Lanczos-3": (kernels.lanczos3, 3.0),
    }
    frequencies = np.linspace(0.0, 0.5, 401)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for label, (kernel, support) in definitions.items():
        values = kernel(x)
        axes[0].plot(x, values, label=label)
        inside = np.abs(x) <= support
        # For symmetric real kernels, the Fourier transform is a cosine integral.
        response = np.array([
            np.sum(values[inside] * np.cos(2.0 * np.pi * f * x[inside])) * dx
            for f in frequencies
        ])
        dc = response[0]
        if abs(dc) > 1e-12:
            response = response / dc
        axes[1].plot(frequencies, np.abs(response), label=label)
        step = np.cumsum(values) * dx
        axes[2].plot(x, step, label=label)

    axes[0].set(title="Spatial reconstruction kernels", xlabel="Distance (source pixels)", ylabel="K(x)")
    axes[1].set(title="Numerical magnitude frequency response", xlabel="Cycles per source pixel", ylabel="|H(f)|")
    axes[2].set(title="Numerical step response", xlabel="Position (source pixels)", ylabel="Cumulative response")
    axes[1].set_xlim(0.0, 0.5)
    axes[2].set_xlim(-3.5, 3.5)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    target = Path(__file__).resolve().parent / "kernel_diagnostics.png"
    fig.savefig(target, dpi=180)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
