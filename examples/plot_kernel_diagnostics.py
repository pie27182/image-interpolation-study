"""Построить графики интерполяционных ядер (альтернатива: image-interpolation plot-kernels)."""
from pathlib import Path

from interpolation_study.visualization import plot_kernel_diagnostics


def main() -> None:
    target = Path(__file__).resolve().parent / "kernel_diagnostics.png"
    print(f"Графики сохранены: {plot_kernel_diagnostics(target)}")


if __name__ == "__main__":
    main()
