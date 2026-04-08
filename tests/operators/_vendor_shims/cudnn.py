raise ImportError(
    "Disable Python cuDNN frontend during operator tests so flashinfer falls back "
    "to its non-cuDNN paths in this container."
)
