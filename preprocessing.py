import os

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = "data/WBC_bak/Dataset_1/"
    for i in range(1, 301):
        img_path = os.path.join(path, f"{i:03d}.bmp")
        seg_path = os.path.join(path, f"{i:03d}.png")
        img = plt.imread(img_path)
        seg = plt.imread(seg_path)
        assert img.shape[:2] == seg.shape
        if img.shape != (120, 120, 3):
            print(img.shape)
        
        labs = np.unique(seg)
        for lab, val in enumerate(labs):
            if lab == 0:
                mask = (seg != val)
            else:
                mask = (seg == val)
            # create directory if not exists
            save_path = os.path.join("data/wbc_1", f"{lab}/{i:03d}")
            os.makedirs(save_path, exist_ok=True)

            # save image
            plt.imsave(os.path.join(save_path, "img.png"), img)
            plt.imsave(os.path.join(save_path, "seg.png"), mask, cmap="gray")
