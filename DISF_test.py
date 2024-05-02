# from disf import DISF_Superpixels
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from nilearn import image
# from skimage import img_as_ubyte, color
# from mpl_toolkits.mplot3d import Axes3D

# # Carrega a imagem NIfTI
# img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
# img = img.get_fdata().transpose(2, 0, 1)[::-1].astype("float")  # Transpõe e inverte as fatias
# img = (img - img.min()) / (img.max() - img.min())  # Normaliza a imagem

# # Converte a imagem normalizada para 8-bit unsigned integer
# img = img_as_ubyte(img)

# # Converte para RGB se a imagem ainda não for RGB
# if img.ndim == 2 or (img.ndim == 3 and img.shape[2] != 3):
#     img = color.gray2rgb(img)

# # Dividindo a imagem em topview e sideview
# top_views = [img[i, :, :] for i in range(img.shape[0])]  # Top views são fatias ao longo do eixo Z
# top_border_imgs = np.empty_like(img)  # Array vazio para armazenar as imagens resultantes

# side_views = [img[:, i, :] for i in range(img.shape[1])]  # Side views são fatias ao longo do eixo Y
# side_border_imgs = np.empty_like(img)  # Array vazio para armazenar as imagens resultantes

# num_init_seeds = 8000
# num_final_superpixels = 50

# # Processa cada fatia de top view com DISF_Superpixels
# for i, top_view in enumerate(top_views):
#     label_img, border_img = DISF_Superpixels(top_view, num_init_seeds, num_final_superpixels)
#     # Copia border_img para todos os três canais
#     top_border_imgs[i, :, :] = np.stack([border_img]*3, axis=-1)  # Empilha border_img em três canais
#     print(i)

# # Processa cada fatia de side view com DISF_Superpixels
# # for i, side_view in enumerate(side_views):
# #     label_img, border_img = DISF_Superpixels(side_view, num_init_seeds, num_final_superpixels)
# #     # Copia border_img para todos os três canais
# #     side_border_imgs[:, i, :] = np.stack([border_img]*3, axis=-1)  # Empilha border_img em três canais

# # Exibindo a primeira fatia de cada resultado para verificação
# # fig = plt.figure(figsize=(10, 5))
# # fig.add_subplot(1, 2, 1)
# # plt.imshow(top_border_imgs[0], cmap='gray', vmin=0, vmax=255)
# # plt.title('Top View Border Image')

# # fig.add_subplot(1, 2, 2)
# # plt.imshow(side_border_imgs[:, 0, :], cmap='gray', vmin=0, vmax=255)
# # plt.title('Side View Border Image')
# # plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Supondo que você queira visualizar um dos canais, como todos são iguais, podemos escolher o primeiro.
# # Isso porque cada canal de top_border_imgs será essencialmente o mesmo se você tiver copiado os valores de `border_img` em todos os três canais.
# voxels = (top_border_imgs[:,:,:,0] > 0)  # Seleciona voxels acima de um limiar, aqui simplificado para maior que zero.

# # Configurar cores: 'gray' para os voxels ativos
# colors = np.empty(voxels.shape, dtype=object)
# colors[voxels] = 'gray'

# # Plotando os voxels
# ax.voxels(voxels, facecolors=colors, edgecolor='k')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.title('3D Visualization of Top View Image')
# plt.show()









from disf import DISF_Superpixels
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from nilearn import image
from skimage import img_as_ubyte, color

# Carrega a imagem NIfTI
img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
img = img.get_fdata().transpose(2, 0, 1)[::-1].astype("float")  # Transpõe e inverte as fatias
img = (img - img.min()) / (img.max() - img.min())  # Normaliza a imagem

# Dividindo a imagem em topview e sideview
top_views = [img[i, :, :] for i in range(img.shape[0])]  # Top views são fatias ao longo do eixo Z
top_views_rgb = [color.gray2rgb(view) for view in top_views]  # Converte top views para RGB

side_views = [img[:, i, :] for i in range(img.shape[1])]  # Side views são fatias ao longo do eixo Y
side_views_rgb = [color.gray2rgb(view) for view in side_views]  # Converte side views para RGB

num_init_seeds = 8000
num_final_superpixels = 50

# Processa cada fatia de top view com DISF_Superpixels
top_border_imgs = []
for top_view_rgb in top_views_rgb:
    top_view_rgb_int = img_as_ubyte(top_view_rgb)  # Converte para inteiro de 8 bits
    top_view_rgb_int_contig = np.ascontiguousarray(top_view_rgb_int, dtype=np.int32)  # Garante contiguidade e tipo de dados
    _, border_img = DISF_Superpixels(top_view_rgb_int_contig, num_init_seeds, num_final_superpixels)
    top_border_imgs.append(border_img)

# Processa cada fatia de side view com DISF_Superpixels
side_border_imgs = []
for side_view_rgb in side_views_rgb:
    side_view_rgb_int = img_as_ubyte(side_view_rgb)  # Converte para inteiro de 8 bits
    side_view_rgb_int_contig = np.ascontiguousarray(side_view_rgb_int, dtype=np.int32)  # Garante contiguidade e tipo de dados
    _, border_img = DISF_Superpixels(side_view_rgb_int_contig, num_init_seeds, num_final_superpixels)
    side_border_imgs.append(border_img)

# Empilhando as imagens de borda sem transformar em RGB novamente
top_border_imgs = np.stack(top_border_imgs, axis=0)
side_border_imgs = np.stack(side_border_imgs, axis=1)

# Exibindo a primeira fatia de cada resultado para verificação
fig = plt.figure(figsize=(10, 5))
fig.add_subplot(1, 2, 1)
plt.imshow(top_border_imgs[0], cmap='gray', vmin=0, vmax=255)
plt.title('Top View Border Image')

fig.add_subplot(1, 2, 2)
plt.imshow(side_border_imgs[:, 0, :], cmap='gray', vmin=0, vmax=255)
plt.title('Side View Border Image')
plt.show()

