from PIL import Image
import os
from skimage import segmentation, io
import numpy as np

def process_and_save_segmentation_borders(input_path, output_path):
    """
    Carrega uma imagem segmentada do caminho fornecido, encontra as bordas da segmentação,
    e salva a imagem resultante das bordas no caminho de saída especificado.
    
    Args:
    input_path (str): Caminho para o arquivo de imagem segmentada.
    output_path (str): Caminho para salvar a imagem das bordas da segmentação.
    """
    # Carregar imagem usando Pillow
    img = Image.open(input_path)
    
    # Converter a imagem PIL para um array NumPy
    img_array = np.array(img)
    
    # Encontrar as bordas dos segmentos
    borders = segmentation.find_boundaries(img_array, mode='thick')
    
    # Converter as bordas de booleano para uint8 (0 ou 255)
    borders_uint8 = (borders * 255).astype(np.uint8)
    
    # Salvar a imagem das bordas usando skimage.io
    io.imsave(output_path, borders_uint8)

# Exemplo de uso:
# Defina os caminhos de entrada e saída
input_path = './top_view/output_63.pgm'
output_path = './top_view_/borders_output.png'

# Chame a função para processar e salvar as bordas
# process_and_save_segmentation_borders(input_path, output_path)


def convert_pgm_to_png(source_directory: str, source_filename: str, target_directory: str, target_filename: str):
    """
    Lê uma imagem .pgm, converte para .png e salva no diretório especificado.

    Parâmetros:
    - source_directory (str): Diretório onde está localizado o arquivo .pgm.
    - source_filename (str): Nome do arquivo .pgm.
    - target_directory (str): Diretório onde o arquivo .png será salvo.
    - target_filename (str): Nome do arquivo .png que será salvo.

    Retorna:
    - None
    """
    # Caminho completo da imagem de origem.
    source_path = os.path.join(source_directory, source_filename + '.pgm')

    # Carrega a imagem .pgm.
    img = Image.open(source_path)

    # Verifica se o diretório de destino existe, caso contrário, cria o diretório.
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Caminho completo da imagem de destino.
    target_path = os.path.join(target_directory, target_filename + '.png')
    

    # Salva a imagem no formato .png.
    img.save(target_path, 'PNG')

    print(f"Imagem convertida e salva como {target_path}")

# Exemplo de uso:
convert_pgm_to_png("./top_view/", "output_63", "./top_view_", "imagem_convertida2")
