import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

# Carregando os dados
data = np.load('segl.npy')

# Função para plotar uma fatia
def plot_slice(slice_index):
    plt.imshow(data[slice_index, :, :], cmap='gray')
    plt.title(f'Visualização da Fatia {slice_index}')
    plt.axis('off')
    plt.show()

# Criando um controle deslizante para selecionar a fatia
interact(plot_slice, slice_index=IntSlider(min=0, max=data.shape[0] - 1, step=1, value=data.shape[0] // 2))
