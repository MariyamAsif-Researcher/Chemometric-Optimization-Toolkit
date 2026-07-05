import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("data/GCC_DEMO.csv")
responses = ["TPC","TFC","DPPH","ABTS","MCA"]
for variable in responses:
plt.figure(figsize=(7,5))

plt.hist(data[variable])

plt.title(variable + " Distribution")

plt.xlabel(variable)

plt.ylabel("Frequency")

plt.savefig("figures/" + variable + "_histogram.png")

plt.close()
