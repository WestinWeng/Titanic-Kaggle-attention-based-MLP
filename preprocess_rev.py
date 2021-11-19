#author Weining Weng
#mail weiningweng1999@gmail.com
#date 2021-11-10 19-18
#kaggle dataset: Titanic preprocess

import os
from scipy import io
import time,datetime
import csv
import numpy as np
from numpy.core.fromnumeric import searchsorted
from tqdm import tqdm
import random

def pro_generate(c1,c2,c3,c4,c5):
    if c1+c2+c3+c4+c5==0:
        c1,c2,c3,c4,c5=20,20,20,20,20
    sum=c1+c2+c3+c4+c5
    ages=np.zeros((5))
    for rand_i in range(1):
        y=random.uniform(0,1)
        if y<=(c1/sum):
            ages[0]=ages[0]+1
        elif y<=((c1+c2)/sum):
            ages[1]=ages[1]+1
        elif y<=((c1+c2+c3)/sum):
            ages[2]=ages[2]+1
        elif y<=((c1+c2+c3+c4)/sum):
            ages[3]=ages[3]+1
        elif y<=1:
            ages[4]=ages[4]+1
    t=[c1,c2,c3,c4,c5]
    return int(np.argmax(ages))+5
        
def certain_probability(data):
    outs=data
    check_fea=np.zeros((data.shape[0],4))
    for i in range(0,data.shape[0]):
        check_fea[i,0:4]=data[i,1:5]
    for i in range(0,data.shape[0]):
        if data[i,5]==1 and data[i,6]==1:
            c1,c2,c3,c4,c5=0,0,0,0,0
            for j in range(0,data.shape[0]):
                re=(check_fea[i,:]==check_fea[j,:])
                if re.all():
                    if data[j,5]==1 and data[j,6]==1 and data[j,7]==1:
                        continue
                    else:
                        if data[j,5]==1:
                            c1=c1+1
                        elif data[j,6]==1:
                            c2=c2+1
                        elif data[j,7]==1:
                            c3=c3+1
                        elif data[j,8]==1:
                            c4=c4=1
                        elif data[j,9]==1:
                            c5=c5+1
            age=pro_generate(c1,c2,c3,c4,c5)
            outs[i,5:10]=0
            outs[i,age]=1
    return outs               

def soc_search(name):
    if name.find('Capt')!=-1 or name.find('Col')!=-1 or name.find('Dr')!=-1 or name.find('Major')!=-1 or name.find('Rev')!=-1 or name.find('Master')!=-1 or name.find('Jonkheer')!=-1:
        pos=0
        return pos
    elif name.find('Sir')!=-1 or name.find('the Countess')!=-1 or name.find('Don')!=-1 or name.find('Dona')!=-1 or name.find('Lady')!=-1:
        pos=1
        return pos
    else:
        pos=2
        return pos


path_train='total.csv'
csv_file=csv.reader(open(path_train,'r'))
rows=[row for row in csv_file]
total_length=len(rows)-1
data=np.zeros((total_length,19),dtype=int)
for i in range(1,total_length+1):
    data[i-1,0]=rows[i][0]  #passenger id
    data[i-1,int(rows[i][2])]=1  #pclass
    if str(rows[i][4])=='male':  #gender
        data[i-1,4]=1
    if rows[i][5]=="":  #age
        data[i-1,5:10]=1
    else:
        if float(rows[i][5])<=10:  
            data[i-1,5]=1
        elif float(rows[i][5])<=20: 
            data[i-1,6]=1
        elif float(rows[i][5])<=40:
            data[i-1,7]=1
        elif float(rows[i][5])<=50:
            data[i-1,8]=1
        elif float(rows[i][5])<=100:
            data[i-1,9]=1        
    if int(rows[i][6])>0: #Sibsp
        data[i-1,10]=1
    if int(rows[i][7])>0: #Patents & children
        data[i-1,11]=1
    if rows[i][11]=="":   #port
            data[i-1,12]=1
    else:
        if str(rows[i][11])=='C':
            data[i-1,12]=1
        elif str(rows[i][11])=='S':
            data[i-1,13]=1
        elif str(rows[i][11])=='Q':
            data[i-1,14]=1
    name=str(rows[i][3])
    socpos=soc_search(name)
    print(socpos)
    data[i-1,15+socpos]=1
    if rows[i][1]=="":
        data[i-1,18]=3
    else:
        data[i-1,18]=int(rows[i][1])


data_new=certain_probability(data)

np.savetxt("preprocess_containment.csv",data_new,delimiter=',') 