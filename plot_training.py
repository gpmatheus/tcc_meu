
import json

with open('first-run/n0-model_info.json', 'r') as json_data:
    d = json.load(json_data)
    json_data.close()
    validmse = d['validmse']
    trainmse = d['trainingmse']

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6))

print(len(trainmse), len(validmse))

plt.plot(np.array(list(range(len(trainmse))))[::10], trainmse[::10], color='blue', label='Erro do Treinamento')
plt.plot(np.array(list(range(len(validmse))))[::10], validmse[::10], color='red', label='Erro de Validação')

plt.title("Treinamento e Validação")
# plt.subtitle(f'MSE: {mse} | RMSE: {rmse}')
plt.xlabel("Épocas")
plt.ylabel("Erro (mse)")
plt.legend() 

plt.show()