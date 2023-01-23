import csv
import glob

import numpy as np
from matplotlib import pyplot as plt

filenames = glob.glob("analysis/*.csv")
for i, filename in enumerate(filenames):
    print(f"{i}: {filename}")

filename_index: int = int(input("\n描画するファイル番号："))
epoch: int = int(input("描画開始するエポック数："))

loss_list: list[float] = []
val_loss_list: list[float] = []
with open(filenames[filename_index]) as f:
    reader = csv.reader(f)
    for _ in range(epoch):
        next(reader)
        loss_list.append(np.nan)
        val_loss_list.append(np.nan)
    for row in reader:
        loss_list.append(float(row[1]))
        val_loss_list.append(float(row[2]))

plt.figure()
plt.plot(loss_list)
plt.plot(val_loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig("./analysis/history_loss.png")
