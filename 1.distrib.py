# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:47:33 2017

@author: Administrator
"""

import scipy.stats as st
import numpy as np
import  matplotlib.pyplot as plt


lambd = 0.5
x=np.arange(0,15,0.1)
y=lambd*np.exp(-lambd*x)
plt.plot(x,y)
plt.title('Exponential')
plt.xlabel('x')
plt.ylabel('pdf')
plt.show()





mu=0
sigma=1
x=np.arange(-5,5,0.1)
y1 = st.norm.pdf(x,0,1)
y2 = st.laplace.pdf(x,0,1)
plt.plot(x,y1,'r')
plt.plot(x,y2,'bo')
plt.title('Normal & Laplace')
plt.xlabel('x')
plt.ylabel('pdf')
plt.legend(['Normal:red line', 'Laplace:blue circles'], loc = 0)# make legend
plt.show()


x=np.arange(0,15,0.1)
y1 = st.gamma.pdf(x,1,0.5)
y2 = st.gamma.pdf(x,2,0.5)
y3 = st.gamma.pdf(x,3,0.5)
y4 = st.gamma.pdf(x,5,2)
plt.plot(x,y1,'r')
plt.plot(x,y2,'g.')
plt.plot(x,y3,'b--')
plt.plot(x,y4,'y:')
plt.title('Gamma')
plt.xlabel('x')
plt.ylabel('pdf')
plt.legend(['a=1,b=0.5', 'a=2,b=0.5', 'a=3,b=0.5', 'a=5,b=2'], loc = 0)# make legend
plt.show()
