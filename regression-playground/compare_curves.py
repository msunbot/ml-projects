import json, matplotlib.pyplot as plt

hs = json.load(open("figs/hist_scratch.json"))
ht = json.load(open("figs/hist_torch.json"))

plt.figure()
plt.plot(hs["va_r2"], label="NumPy Val R2")
plt.plot(ht["va_r2"], label="Torch Val R2")
plt.xlabel("epoch"); plt.ylabel("R2"); plt.legend(); plt.title("Val R2: NumPy vs Torch")
plt.show()