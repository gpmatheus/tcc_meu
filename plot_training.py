
# import json

# with open('seventh-run(same_as_fourth-L2=5e-5-lr=5e-5)/n0-model_info.json', 'r') as json_data:
#     d = json.load(json_data)
#     json_data.close()
#     validmse = d['validmse']
#     trainmse = d['trainingmse']

# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(6, 6))

# print(len(trainmse), len(validmse))

# plt.plot(np.array(list(range(len(trainmse))))[::10], trainmse[::10], color='blue', label='Erro do Treinamento')
# plt.plot(np.array(list(range(len(validmse))))[::10], validmse[::10], color='red', label='Erro de Validação')

# plt.title("Treinamento e Validação")
# # plt.subtitle(f'MSE: {mse} | RMSE: {rmse}')
# plt.xlabel("Épocas")
# plt.ylabel("Erro (mse)")
# plt.legend() 

# plt.show()


import json

with open('update/img-previous_img/seventh-run(same_as_fourth-L2=5e-5-lr=5e-5)/n0-model_info.json', 'r') as json_data2, \
    open('original/multiple-files/second-run/n0-model_info.json', 'r') as json_data1:
    d1 = json.load(json_data1)
    d2 = json.load(json_data2)
    json_data1.close()
    json_data2.close()
    validmse1 = d1['validmse']
    trainmse1 = d1['trainingmse']
    validmse2 = d2['validmse']
    trainmse2 = d2['trainingmse']

import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(6, 6))

print(1, len(trainmse1), len(validmse1))

f, axarr = plt.subplots(1, 2, figsize=(8, 4))

axarr[0].plot(np.array(list(range(len(trainmse1))))[::10], trainmse1[::10], color='blue', label='Erro do Treinamento')
axarr[0].plot(np.array(list(range(len(validmse1))))[::10], validmse1[::10], color='red', label='Erro de Validação')
axarr[0].set_title("Modelo sem diferença nas imagens")
axarr[0].set_xlabel("Épocas")
axarr[0].set_ylabel("Erro (mse)")

axarr[1].plot(np.array(list(range(len(trainmse2))))[::10], trainmse2[::10], color='blue', label='Erro do Treinamento')
axarr[1].plot(np.array(list(range(len(validmse2))))[::10], validmse2[::10], color='red', label='Erro de Validação')
axarr[1].set_title("Modelo com diferença nas imagens")
axarr[1].set_xlabel("Épocas")
axarr[1].set_ylabel("Erro (mse)")

plt.legend()

# plt.title("Treinamento e Validação")
# # plt.subtitle(f'MSE: {mse} | RMSE: {rmse}')
# plt.xlabel("Épocas")
# plt.ylabel("Erro (mse)")
# plt.legend() 

plt.show()