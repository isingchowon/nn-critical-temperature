# 3D Ising model

`ising_multiseed.py` generates the data (C code: Metropolis + Wolff, compiled automatically), trains the three
networks for every window and seed, and prints the summary table and the figure.

| Network | Input |
|---|---|
| `cnn_per_size` | raw spins; standard CNN (two 3x3x3 convolutions, 32 channels, periodic padding, global average pooling, MLP 32-32-1), one network per L |
| `cnn_shared` | raw spins; the same CNN trained once on all sizes |
| `cnn_rescaled` | scale-invariant CNN: flip-odd local field, lattice mean M, only \|M\| L^(beta/nu) reaches the MLP 1-16-1 |

Baselines: Binder cumulant and <|m|> L^(beta/nu) crossings, computed from the same held-out configurations.
Results go to `MyDrive/ising_thesis/ising_multiseed_results.json` and `ising_multiseed.png`.
