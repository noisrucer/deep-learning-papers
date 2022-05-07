import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fig.suptitle('Figure Title')

ax.plot([1,3,2], label='legend')
ax.legend()

ax.text(x=1,y=2, s='Animal: 0.99',
        fontsize=10,
        color='red')

plt.show()
