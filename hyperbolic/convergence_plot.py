import matplotlib.pyplot as plt
import numpy as np

H = [np.float64(0.5656854249492382), np.float64(0.2828427124746193), np.float64(0.1131370849898477), np.float64(0.05656854249492385)]
P1 = [np.float64(0.09797914476696379), np.float64(0.026126944724678327), np.float64(0.0045761986509646035), np.float64(0.001104560220154185)]
P2 = [np.float64(0.04512295796747249), np.float64(0.01310744559342181), np.float64(0.0021938501318828287), np.float64(0.000546206714456754)]
K1 = [0.47454967598173076, 0.32235563951757173, 0.16867138852810326, 0.11322085346260174]
K2 = [0.33546360662301977, 0.18992878274174194, 0.06386490289913581, 0.030356326618873292]


plt.loglog(H, P1, "o--", color="black", label="Poincaré k=1")
plt.loglog(H, P2, "<--", color="black", label="Poincaré k=2")
plt.loglog(H, K1, "o--", color="gray", label="Klein k=1")
plt.loglog(H, K2, "<--", color="gray", label="Klein k=2")
plt.ylabel("$\log \epsilon$")
plt.xlabel("$\log h$")
plt.legend()

plt.show()