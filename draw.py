import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

epochs = [1,2,3,4,5]

loss = [1180.8304, 358.9501, 144.8750, 65.1066, 22.1549]
em = [0.8382, 0.8319, 0.8358, 0.8355, 0.8431]
fig, axis = plt.subplots(1,2)

#plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

axis[0].plot(epochs, loss, 's-')
axis[0].set_title("Loss")
axis[0].set_xlabel("epoch")
axis[1].plot(epochs, em, 's-')
axis[1].set_title("Exact Match")
axis[1].set_xlabel("epoch")
plt.show()
plt.savefig("./qa_curves.png")
