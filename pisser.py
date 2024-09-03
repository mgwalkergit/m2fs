# importing matplotlib package
import matplotlib.pyplot as plt
 
# importing the numpy package
import numpy as np
 
# Storing set of values in
# names, x, height,
# error and colors for plotting the graph
names= ['Bijon', 'Sujit', 'Sayan', 'Saikat']
x=np.arange(4)
marks=[ 60, 90, 55, 46]
error=[ 11, 15, 5, 9]
colors = ['red', 'green', 'blue', 'navy']
 
# using tuple unpacking
# to grab fig and axes
fig, ax = plt.subplots()
 
# plotting the bar plot
ax.bar(x, marks, alpha = 0.5,
       color = colors)
 
# Zip function acts as an
# iterator for tuples so that
# we are iterating through
# each set of values in a loop
for pos, y, err, colors in zip(x, marks,
                               error, colors):
   
    ax.errorbar(pos, y, err, lw = 2,
                capsize = 4, capthick = 4,
                color = colors,markersize=1)
    
