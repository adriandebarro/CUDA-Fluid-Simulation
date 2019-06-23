import matplotlib.pyplot as plt
import csv
import numpy as np

y = []

with open('results.csv','r') as csvfile:
    nums = csv.reader(csvfile, delimiter=',')
    for num in nums:
        y.append(num)

config = ''
for item in y:
	if len(item) > 1:
		data = [float(i) for i in item[0:-1]]
		x = np.arange(len(item)-1)
		x = x + 6 - len(item) + 1
		plt.plot(x, data, label=config)
	else:
		config = item[0]

plt.xticks(np.arange(6), ('32', '16', '8', '4', '2', '1'))
plt.xlabel('tidsy')
plt.ylabel('time')
plt.legend()
plt.show()
