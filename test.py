import numpy as np
import xlrd
import matplotlib.pyplot as plt 

file=xlrd.open_workbook('save_accu.xlsx')
sheet1=file.sheets()[0]
column=sheet1.ncols
row=sheet1.nrows
draw=np.zeros((row,column))
for i in range(0,row):
    for j in range(0,column):
        draw[i,j]=float(sheet1.row(i)[j].value)
plt.plot(draw[0,:],draw[1,:],label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim(0,1.2)
plt.grid()
plt.legend()
plt.show()
