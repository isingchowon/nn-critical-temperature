
# nn-critical-temperature

Code for the BSc thesis *Finite-Size Scaling of Neural-Network Classifier Outputs:
Critical Temperatures of the 3D Ising Model and the Sherrington–Kirkpatrick Spin Glass*
(Chowon Lee, Hanyang University, 2027; advisor Prof. Park Moon Jip).

| Script | Produces |
|---|---|
| `ISING_SCRIPT_NAME.py` | 3D Ising data, CNNs, Table II, Figure I, exponent test |
| `sk_multiseed_N256.py` | SK data (C parallel tempering), all SK networks, Table III, Figure II |

Both scripts run on Google Colab with a T4 GPU. Paste a script into one cell and run it.
Results are saved to Google Drive; five training seeds (0–4) are used throughout.
