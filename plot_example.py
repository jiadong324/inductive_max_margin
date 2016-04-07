import matplotlib.pylab as plt

x = [1,2,3,4]
y = [0.5,0.6,0.4,0.7]

plt.plot(x,y,'bo-')
plt.axhline(y=0.5,c='r',ls='--')
plt.ylim([0,max(y)+0.1])
plt.show()