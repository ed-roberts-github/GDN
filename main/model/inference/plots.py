import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.chdir("/Users/edroberts/Desktop/im_gen/training_data/testing/general_metrics/")
df = pd.read_csv("metrics_28x28v3.csv", sep=",", header=None)
r, purity, p_err, completeness, c_err = df.values

df14 = pd.read_csv("metrics_14x14v3.csv", sep=",", header=None)
r14, purity14, p_err14, completeness14, c_err14 = df14.values

df7 = pd.read_csv("metrics_7x7v5.csv", sep=",", header=None)
r7, purity7, p_err7, completeness7, c_err7 = df7.values


plt.errorbar(r, completeness, c_err, color='c',label = '28x28', capsize=(2))
plt.errorbar(r14, completeness14, c_err14, color='g',label = '14x14', capsize=(2))
plt.errorbar(r7, completeness7, c_err7, color='r',label = '7x7', capsize=(2))
plt.title("Completeness over truth radius")
plt.xlabel("Truth radius r_tp (pixels)")
plt.ylabel("Completeness")
plt.legend()
plt.savefig("Completenessv6.png")
plt.close()


# plt.plot(r, purity,  color='c',label = '28x28')
# plt.plot(r14, purity14,  color='g',label ='14x14' )
# plt.plot(r7, purity7,  color='r',label ='7x7' )
plt.errorbar(r, purity, p_err,  color='c',label = '28x28', capsize=(2))
plt.errorbar(r14, purity14, p_err14,  color='g',label ='14x14', capsize=(2) )
plt.errorbar(r7, purity7, p_err7,  color='r',label ='7x7', capsize=(2) )
plt.title("Purity over truth radius")
plt.xlabel("Truth radius r_tp (pixels)")
plt.ylabel("Purity")
plt.legend()
plt.savefig("Purityv6.png")
plt.close()


os.chdir("/Users/edroberts/Desktop/im_gen/training_data/testing/snr/")
df = pd.read_csv("SNR_sweepv3.csv", sep=",", header=None)
snr, purity, purity_err, completeness ,completeness_err= df.values

plt.errorbar(snr, completeness,completeness_err,capsize=(3), color='c')
# plt.plot(snr, completeness, color='c')
plt.title("Completeness over SNR range for (28x28) model")
plt.xlabel("SNR")
plt.ylabel("Completeness")
plt.savefig("Completeness_SNRv6.png")
plt.close()


# plt.plot(snr, purity,  color='c')
plt.errorbar(snr,purity,purity_err,capsize=(3), color='c')
plt.title("Purity over SNR range for (28x28) model")
plt.xlabel("SNR")
plt.ylabel("Purity")
plt.savefig("Purity_SNRv6.png")
plt.close()



# import plotly.graph_objs as go 
# layout = go.Layout(
#     title='Completeness vs SNR range',
#     xaxis=dict(title='SNR range'),
#     yaxis=dict(title='Completeness', range=[0, 1]),
#     font=dict(size=12, family='serif'),
#     legend=dict(font=dict(size=10))
# )

# trace1 = go.Scatter(
#     x=snr,
#     y=completeness,
#     mode='lines+markers',
#     name='Data',
#     line=dict(color='blue', width=2),
#     marker=dict(size=6),
#     error_y=dict(
#         type='data',
#         array=completeness_err,
#         visible=True,
#         color='gray',
#         thickness=2,
#         width=3,
#     ),
#     hovertemplate='SNR range: %{x:.2f}<br>Completeness: %{y:.2f}<extra></extra>'
# )

# fig = go.Figure(data=[trace1], layout=layout)
# fig.show()
