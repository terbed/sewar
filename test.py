from full_ref import ssim_map
import PIL.Image
import numpy as np
from matplotlib import pyplot as plt

gt = PIL.Image.open("test_samples/ref.png")
o3 = PIL.Image.open("test_samples/out3.png")

ssimmap, _ = ssim_map(np.array(gt),  np.array(o3))

print(np.min(ssimmap), np.max(ssimmap))
print(np.mean(ssimmap))

inv = 1-ssimmap
plt.figure(figsize=(12, 12))
plt.imshow(inv, cmap="jet")
plt.colorbar()


inv_thrs = inv.copy()
inv_thrs[inv_thrs < 0.2] = np.nan

plt.figure(figsize=(12, 12))
plt.imshow(inv_thrs, cmap="jet")
plt.colorbar()
plt.show()
