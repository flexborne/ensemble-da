"""Toy forecast model for the examples: periodic 1-D advection-diffusion, exact spectral propagator.

    dc/dt + u dc/dx = kappa d2c/dx2   on [0, length), n cells

It is deliberately not part of the ensemble_da package: the assimilation module only consumes
the ensembles a model produces.
"""
from __future__ import annotations

import numpy as np


class Transport1D:
    def __init__(self, n=200, length=1.0, diffusivity=1e-4):
        self.n, self.length, self.diffusivity = n, length, diffusivity
        self.x = (np.arange(n) + 0.5) * length / n
        self._k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)

    def propagate(self, c, dt, velocity):
        """Advance a field c (n,) or an ensemble (n, M) by dt; velocity is a scalar or one per member."""
        c = np.asarray(c, dtype=float)
        k = self._k[:, None] if c.ndim == 2 else self._k
        factor = np.exp((-1j * k * np.asarray(velocity, dtype=float) - self.diffusivity * k ** 2) * dt)
        return np.real(np.fft.ifft(np.fft.fft(c, axis=0) * factor, axis=0))

    def random_field(self, corr_length, rng, size=None):
        """Zero-mean, unit-variance periodic Gaussian random field(s) with a Gaussian correlation
        of length corr_length: (n,) or (n, size)."""
        amplitude = np.exp(-0.25 * (self._k * corr_length) ** 2)
        noise = rng.standard_normal((self.n,) if size is None else (self.n, size))
        g = np.real(np.fft.ifft(np.fft.fft(noise, axis=0) * (amplitude if size is None else amplitude[:, None]), axis=0))
        return g / g.std()
