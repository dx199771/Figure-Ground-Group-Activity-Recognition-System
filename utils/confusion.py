"""
    Author: Xu Dong
    Student Number: 200708160
    Email: x.dong@se20.qmul.ac.uk

    School of Electronic Engineering and Computer Science
    Queen Mary University of London, UK
    London, UK
"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# input experiment results here
array = []

x_axis = ["Ball out of play","Clearance","Corner","Substitution","Yellow card","Throw-in"]
y_axis = ["Ball out of play","Clearance","Corner","Substitution","Yellow card","Throw-in"]

df_cm = pd.DataFrame(array, x_axis, y_axis)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.5) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 22},fmt='.1%',cmap='Blues') # font size

plt.show()