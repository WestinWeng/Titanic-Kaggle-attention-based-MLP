import numpy as np
import xlrd
import matplotlib.pyplot as plt 

loss=np.load('loss.npy')
x=range(0,100)
plt.plot(x,loss,label='Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim(0,1.2)
plt.grid()
plt.legend()
plt.show()
