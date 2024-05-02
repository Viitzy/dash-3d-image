from idisf import iDISF_scribbles
import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash import html
from dash import dcc
import plotly.graph_objects as go
from skimage import data, img_as_ubyte, segmentation, measure
from dash_canvas.utils import array_to_data_url
import plotly.graph_objects as go
import plot_common
import image_utils
import numpy as np
from nilearn import image
import nibabel as nib
import plotly.express as px
import shape_utils
from sys import exit
import io
import base64
import skimage
import time
import os
import re

import matplotlib.pyplot as plt
import torch

from PIL import Image
from io import BytesIO


DEBUG_MASK = False # Variavel para debugar, nao sei como funciona
DEFAULT_STROKE_COLOR = px.colors.qualitative.Light24[0] # Nao  sei do que se trata
DEFAULT_STROKE_WIDTH = 5 # Nao  sei do que se trata

# escalas das imagens topside e side, largura e altura devem ser as mesmas
hwscales = [(2, 2), (2, 2)] 

# the number of dimensions displayed
NUM_DIMS_DISPLAYED = 2  # top and side, numeros de dimensoes

#cor do trianglinho que aparece do lado da imagem
INDICATOR_COLOR = "DarkOrange"
DISPLAY_BG_COLOR = "darkgrey"

# A string, if length non-zero, saves superpixels to this file and then exits
#string que salva o superpixel
SAVE_SUPERPIXEL = os.environ.get("SAVE_SUPERPIXEL", default="")
# A string, if length non-zero, loads superpixels from this file
#string que carrega a segmentacao ja para exibicao
LOAD_SUPERPIXEL = os.environ.get("LOAD_SUPERPIXEL", default="")
# If not "0", debugging mode is on.
#variavel que define se o codigo esta em debug, nunca utilizei
DEBUG = os.environ.get("DEBUG", default="0") != "0"

#funcao para printar algo caso esteja em modo debug
def PRINT(*vargs):
    if DEBUG:
        print(*vargs)


#funcao que supostamente cria as imagens iniciais, cria imagens vazias para ser preenchidas com 
#a segmentacao apos ela ter sido feita, nao tenho certeza disso
def make_default_figure(
    images=[],
    stroke_color=DEFAULT_STROKE_COLOR,
    stroke_width=DEFAULT_STROKE_WIDTH,
    img_args=dict(layer="above"),
    width_scale=1,
    height_scale=1,
):
    fig = plot_common.dummy_fig()
    plot_common.add_layout_images_to_fig(
        fig,
        images,
        img_args=img_args,
        width_scale=width_scale,
        height_scale=height_scale,
        update_figure_dims="height",
    )
    # add an empty image with the same size as the greatest of the already added
    # images so that we can add computed masks clientside later
    mwidth, mheight = [
        max([im[sz] for im in fig["layout"]["images"]]) for sz in ["sizex", "sizey"]
    ]
    fig.add_layout_image(
        dict(
            source="",
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=mwidth,
            sizey=mheight,
            sizing="contain",
            layer="above",
        )
    )
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
            "plot_bgcolor": DISPLAY_BG_COLOR,
        }
    )
    return fig


#Carrega as imagens da tomografia e transforma as mesmas no formato esperado
img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
img = img.get_fdata().transpose(2, 0, 1)[::-1].astype("float")
img = img_as_ubyte((img - img.min()) / (img.max() - img.min()))

#cria um tensor com o mesmo tamanho de img, define o tensor como caso marcado(o pixel) ele será da cor roxa
#apos isso reorganiza a imagem, separando a mesma em lista de listas, para cada fatia da imagem ser uma url de dados
def make_empty_found_segments():
    """ fstc_slices is initialized to a bunch of images containing nothing (clear pixels) """
    found_segs_tensor = np.zeros_like(img)
    # convert to a colored image (but it will just be colored "clear")
    fst_colored = image_utils.label_to_colors(
        found_segs_tensor,
        colormap=["#000000", "#8A2BE2"],
        alpha=[0, 128],
        color_class_offset=0,
    )
    fstc_slices = [
        [
            array_to_data_url(np.moveaxis(fst_colored, 0, j)[i])
            for i in range(np.moveaxis(fst_colored, 0, j).shape[0])
        ]
        for j in range(NUM_DIMS_DISPLAYED)
    ]
    return fstc_slices

#transforma a tomografia em ubyte, e entao cria uma tupla img_slices, seg_slices
#que indica os cortes da imagem e os cortes da segmentacao
#Este bloco cria listas de URLs de dados para cada fatia de img e seg_img 
#(a imagem segmentada). O loop percorre as duas imagens, criando duas visualizações
#para cada imagem: uma "top view" (vista de cima) e uma "side view" (vista lateral),
#convertendo cada fatia da imagem em uma URL de dados para ser utilizada em uma 
#interface web.
seg_img = img_as_ubyte(img)
img_slices, seg_slices = [
    [
        # top
        [array_to_data_url(im[i, :, :]) for i in range(im.shape[0])],
        # side
        [array_to_data_url(im[:, i, :]) for i in range(im.shape[1])],
    ]
    for im in [img, seg_img]
]

# initially no slices have been found so we don't draw anything
# Como nao foi encontrado nenhuma particao da imagem entao é criada as particoes vazias
found_seg_slices = make_empty_found_segments()

# store encoded blank slices for each view to save recomputing them for slices
# containing no colored pixels
# armazena a primeira fatia de cada vista como uma fatia em branco para uso futuro,
# evitando recomputação.
blank_seg_slices = [found_seg_slices[0][0], found_seg_slices[1][0]]

#inicializa o servidor do dash
app = dash.Dash(__name__)
server = app.server

#Cria as visualizacoes topside e side utilizando a funcao make default figure
top_fig, side_fig = [
    make_default_figure(
        images=[img_slices[i][0], seg_slices[i][0]],
        width_scale=hwscales[i][1],
        height_scale=hwscales[i][0],
    )
    for i in range(NUM_DIMS_DISPLAYED)
]

#Cria o layout para a visualizacao 3D
default_3d_layout = dict(
    scene=dict(
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, ticks=""),
        xaxis=dict(visible=True, title="Side View Slice Number"),
        zaxis=dict(visible=True, title="Top View Slice Number"),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        ),
    ),
    height=800,
)

#cria o layout da imagem 3d
def make_default_3d_fig():
    fig = go.Figure(data=[go.Mesh3d()])
    fig.update_layout(**default_3d_layout)
    return fig

# A função make_modal() cria uma janela modal que inicialmente está oculta e pode ser mostrada ao usuário quando necessário.
#exibe o conteudo de assets/howto.md
def make_modal():
    with open("assets/howto.md", "r") as f:
        readme_md = f.read()

    return html.Div(
        id="markdown",
        className="modal",
        style={"display": "none"},
        children=[
            html.Div(
                id="markdown-container",
                className="markdown-container",
                # style={
                #     "color": text_color["light"],
                #     "backgroundColor": card_color["light"],
                # },
                children=[
                    html.Div(
                        className="close-container",
                        children=html.Button(
                            "Close",
                            id="markdown_close",
                            n_clicks=0,
                            className="closeButton",
                            style={"color": "DarkBlue"},
                        ),
                    ),
                    html.Div(
                        className="markdown-text", children=dcc.Markdown(readme_md)
                    ),
                ],
            )
        ],
    )

#cria layout principal da aplicação
app.layout = html.Div(
    id="main",
    children=[
        # Banner display
        html.Div(
            id="banner",
            children=[
                html.Div(
                    html.H1(
                        "3D Image Annotation",
                        id="title",
                        style={
                            "color": "#f9f9f9",
                            "display": "inline-block",
                            "margin": "0",
                        },
                    ),
                ),
                html.Div(
                    html.Button(
                        "Learn more",
                        id="learn-more-button",
                        n_clicks=0,
                        style={"width": "auto"},
                    ),
                ),
                # Adding the modal content here. It is only shown if the show-modal
                # button is pressed
                make_modal(),
            ],
            style={
                "display": "flex",
                "position": "relative",
                "margin": "10px 10px 10px 10px",
            },
        ),
        dcc.Store(id="image-slices", data=img_slices),
        dcc.Store(id="seg-slices", data=seg_slices),
        dcc.Store(
            id="drawn-shapes",
            data=[
                [[] for _ in range(seg_img.shape[i])] for i in range(NUM_DIMS_DISPLAYED)
            ],
        ),
        dcc.Store(id="slice-number-top", data=0),
        dcc.Store(id="slice-number-side", data=0),
        dcc.Store(
            id="undo-data",
            data=dict(
                undo_n_clicks=0,
                redo_n_clicks=0,
                undo_shapes=[],
                redo_shapes=[],
                # 2 arrays, one for each image-display-graph-{top,side}
                # each array contains the number of slices in that image view, and each
                # item of this array contains a list of shapes
                empty_shapes=[
                    [[] for _ in range(seg_img.shape[i])]
                    for i in range(NUM_DIMS_DISPLAYED)
                ],
            ),
        ),
        # In this implementation we want to prevent needless passing of the
        # large image array from client to server, so when "downloaded-button"
        # is clicked, the contents of the "found-segs" store is converted to nii
        # imaging format, converted to base64, and stored in the
        # "found-image-tensor-data" store. When this store's contents are
        # updated, they are stored, decoded, in a Blob and a url is created from
        # the contents of this blob and set as the href of "download-link". Then
        # somehow we need to simulate a "click" on the "download-link". The
        # "found-image-tensor-data" store is necessary because we can only pass
        # base64-encoded data between client and server: we let the browser
        # handle how data from the browser can be written to the client's
        # filesystem.
        html.Div(
            id="loader-wrapper",
            children=[
                # required so callback triggered by writing to "found-image-tensor-data"
                # has an output
                html.Div(id="dummy", style={"display": "none"}),
                html.Div(id="dummy2", style={"display": "none"}, children=",0"),
                # hidden elements so we can show/hide segmentations on 2d and 3d figures
                html.Div(
                    id="show-hide-seg-2d", children="show", style={"display": "none"}
                ),
                html.Div(
                    id="show-hide-seg-3d", children="show", style={"display": "none"}
                ),
                dcc.Loading(
                    id="graph-loading",
                    type="circle",
                    children=[
                        html.A(id="download-link", download="found_image.nii",),
                        # the image data of the found segmentation is stored
                        # here before it is downloaded
                        dcc.Store(id="found-image-tensor-data", data=""),
                        html.Div(
                            children=[
                                html.Button(
                                    "3D View",
                                    id="view-select-button",
                                    n_clicks=0,
                                    style={"width": "25%"},
                                ),
                                html.Button(
                                    "Hide Segmentation",
                                    id="show-seg-check",
                                    n_clicks=0,
                                    style={"width": "25%"},
                                ),
                                html.Button(
                                    "Download Brain Volume",
                                    id="download-brain-button",
                                    style={"width": "auto"},
                                ),
                                html.Button(
                                    "Download Selected Partitions",
                                    id="download-button",
                                    style={"width": "auto"},
                                ),
                                html.Button(
                                    "Undo",
                                    id="undo-button",
                                    n_clicks=0,
                                    style={"width": "12.5%"},
                                ),
                                html.Button(
                                    "Redo",
                                    id="redo-button",
                                    n_clicks=0,
                                    style={"width": "12.5%"},
                                ),
                            ],
                            style={"display": "flex", "margin": "2px 0 2px 0"},
                        ),
                        html.Div(
                            id="2D-graphs",
                            style={
                                "display": "grid",
                                "grid-template-columns": "repeat(2,1fr)",
                                "grid-auto-rows": "auto",
                                "grid-gap": "0 2px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.H6(
                                            "Top View", style={"text-align": "center",}
                                        )
                                    ],
                                    style={
                                        "grid-column": "1",
                                        "grid-row": "1",
                                        "background-color": DISPLAY_BG_COLOR,
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="image-display-graph-top",
                                            figure=top_fig,
                                        )
                                    ],
                                    style={
                                        "grid-column": "1",
                                        "grid-row": "2",
                                        "background-color": DISPLAY_BG_COLOR,
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            id="image-select-top-display",
                                            style={"width": "125px"},
                                        ),
                                        html.Div(
                                            dcc.Slider(
                                                id="image-select-top",
                                                min=0,
                                                max=len(img_slices[0]) - 1,
                                                step=1,
                                                updatemode="drag",
                                                value=len(img_slices[0]) // 2,
                                            ),
                                            style={"flex-grow": "1"},
                                        ),
                                    ],
                                    style={
                                        "grid-column": "1",
                                        "grid-row": "3",
                                        "display": "flex",
                                        "background": "grey",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Side View", style={"text-align": "center"}
                                        )
                                    ],
                                    style={
                                        "grid-column": "2",
                                        "grid-row": "1",
                                        "background-color": DISPLAY_BG_COLOR,
                                    },
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="image-display-graph-side",
                                            figure=side_fig,
                                        )
                                    ],
                                    style={
                                        "grid-column": "2",
                                        "grid-row": "2",
                                        "background-color": DISPLAY_BG_COLOR,
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            id="image-select-side-display",
                                            style={"width": "125px"},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Slider(
                                                    id="image-select-side",
                                                    min=0,
                                                    max=len(img_slices[1]) - 1,
                                                    step=1,
                                                    updatemode="drag",
                                                    value=len(img_slices[1]) // 2,
                                                )
                                            ],
                                            style={"flex-grow": "1"},
                                        ),
                                    ],
                                    style={
                                        "grid-column": "2",
                                        "grid-row": "3",
                                        "display": "flex",
                                        "background": "grey",
                                    },
                                ),
                                # This store has to be put here so dcc.Loading sees that it is updating.
                                dcc.Store(id="found-segs", data=found_seg_slices),
                            ],
                        ),
                        html.Div(
                            id="3D-graphs",
                            children=[
                                dcc.Graph(
                                    "image-display-graph-3d",
                                    figure=make_default_3d_fig(),
                                    config=dict(displayModeBar=False,),
                                )
                            ],
                            style={"display": "none"},
                        ),
                    ],
                ),
            ],
        ),
        dcc.Store(id="fig-3d-scene", data=default_3d_layout,),
        dcc.Store(id="current-render-id", data=0),
        dcc.Store(id="last-render-id", data=0),
    ],
)

#call back respectiva ao botao show segmentation, que mostra a segmentacao ou apaga a mesma
#pode ser que essa funcao altera somente o nome do botao
app.clientside_callback(
    """
function (show_seg_n_clicks) {
    // update show segmentation button
    var show_seg_button = document.getElementById("show-seg-check");
    if (show_seg_button) {
        show_seg_button.textContent = show_seg_n_clicks % 2 ?
            "Show Segmentation" :
            "Hide Segmentation";
    }
    var ret = (show_seg_n_clicks % 2) ? "" : "show";
    return [ret,ret];
}
""",
    [Output("show-hide-seg-2d", "children"), Output("show-hide-seg-3d", "children")],
    [Input("show-seg-check", "n_clicks")],
)


# Este código facilita a interação dinâmica e a atualização das visualizações de 
# imagens médicas em um aplicativo de anotação ou análise de imagem, permitindo que 
# o usuário controle quais fatias são exibidas e se as segmentações devem ser visíveis.
app.clientside_callback(
    """
function(
    image_select_top_value,
    image_select_side_value,
    show_hide_seg_2d,
    found_segs_data,
    image_slices_data,
    image_display_top_figure,
    image_display_side_figure,
    seg_slices_data,
    drawn_shapes_data) {{
    let show_seg_check = show_hide_seg_2d;
    let image_display_figures_ = figure_display_update(
        [image_select_top_value,image_select_side_value],
        show_seg_check,
        found_segs_data,
        image_slices_data,
        [image_display_top_figure,image_display_side_figure],
        seg_slices_data,
        drawn_shapes_data),
        // slider order reversed because the image slice number is shown on the
        // other figure
        side_figure = image_display_figures_[1],
        top_figure = image_display_figures_[0],
        d=3,
        sizex, sizey;
    // append shapes that show what slice the other figure is in
    sizex = top_figure.layout.images[0].sizex,
    sizey = top_figure.layout.images[0].sizey;
    // tri_shape draws the triangular shape, see assets/app_clientside.js
    if (top_figure.layout.shapes) {{
        top_figure.layout.shapes=top_figure.layout.shapes.concat([
            tri_shape(d/2,sizey*image_select_side_value/found_segs_data[1].length,
                      d/2,d/2,'right'),
            tri_shape(sizex-d/2,sizey*image_select_side_value/found_segs_data[1].length,
                      d/2,d/2,'left'),
        ]);
    }}
    sizex = side_figure.layout.images[0].sizex,
    sizey = side_figure.layout.images[0].sizey;
    if (side_figure.layout.shapes) {{
        side_figure.layout.shapes=side_figure.layout.shapes.concat([
            tri_shape(d/2,sizey*image_select_top_value/found_segs_data[0].length,
                      d/2,d/2,'right'),
            tri_shape(sizex-d/2,sizey*image_select_top_value/found_segs_data[0].length,
                      d/2,d/2,'left'),
        ]);
    }}
    // return the outputs
    return image_display_figures_.concat([
    "Slice: " + (image_select_top_value+1) + " / {num_top_slices}",
    "Slice: " + (image_select_side_value+1) + " / {num_side_slices}",
    image_select_top_value,
    image_select_side_value
    ]);
}}
""".format(
        num_top_slices=len(img_slices[0]), num_side_slices=len(img_slices[1])
    ),
    [
        Output("image-display-graph-top", "figure"),
        Output("image-display-graph-side", "figure"),
        Output("image-select-top-display", "children"),
        Output("image-select-side-display", "children"),
        Output("slice-number-top", "data"),
        Output("slice-number-side", "data"),
    ],
    [
        Input("image-select-top", "value"),
        Input("image-select-side", "value"),
        Input("show-hide-seg-2d", "children"),
        Input("found-segs", "data"),
    ],
    [
        State("image-slices", "data"),
        State("image-display-graph-top", "figure"),
        State("image-display-graph-side", "figure"),
        State("seg-slices", "data"),
        State("drawn-shapes", "data"),
    ],
)
#controla os desenhos do usuario
# Esta função é essencial para gerenciar interativamente as anotações nas imagens, 
# permitindo aos usuários desfazer e refazer alterações de forma eficaz e manter a 
# integridade dos dados de anotação em uma interface de usuário reativa e interativa.
app.clientside_callback(
    """
function(top_relayout_data,
side_relayout_data,
undo_n_clicks,
redo_n_clicks,
top_slice_number,
side_slice_number,
drawn_shapes_data,
undo_data)
{
    // Ignore if "shapes" not in any of the relayout data
    let triggered = window.dash_clientside.callback_context.triggered.map(
        t => t['prop_id'])[0];
    if ((triggered === "image-display-graph-top.relayoutData" && !("shapes" in
        top_relayout_data)) || (triggered === "image-display-graph-side.relayoutData"
        && !("shapes" in side_relayout_data))) {
        return [window.dash_clientside.no_update,window.dash_clientside.no_update];
    }
    drawn_shapes_data = json_copy(drawn_shapes_data);
    let ret = undo_track_slice_figure_shapes (
    [top_relayout_data,side_relayout_data],
    ["image-display-graph-top.relayoutData",
     "image-display-graph-side.relayoutData"],
    undo_n_clicks,
    redo_n_clicks,
    undo_data,
    drawn_shapes_data,
    [top_slice_number,side_slice_number],
    // a function that takes a list of shapes and returns those that we want to
    // track (for example if some shapes are to show some attribute but should not
    // be tracked by undo/redo)
    function (shapes) { return shapes.filter(function (s) {
            let ret = true;
            try { ret &= (s.fillcolor == "%s"); } catch(err) { ret &= false; }
            try { ret &= (s.line.color == "%s"); } catch(err) { ret &= false; }
            // return !ret because we don't want to keep the indicators
            return !ret;
        });
    });
    undo_data=ret[0];
    drawn_shapes_data=ret[1];
    return [drawn_shapes_data,undo_data];
}
"""
    % ((INDICATOR_COLOR,) * 2),
    [Output("drawn-shapes", "data"), Output("undo-data", "data")],
    [
        Input("image-display-graph-top", "relayoutData"),
        Input("image-display-graph-side", "relayoutData"),
        Input("undo-button", "n_clicks"),
        Input("redo-button", "n_clicks"),
    ],
    [
        State("slice-number-top", "data"),
        State("slice-number-side", "data"),
        State("drawn-shapes", "data"),
        State("undo-data", "data"),
    ],
)

#Aqui abaixo comeca sequencia de codigos da tentativa de segmentar a imagem baseada ao input
def extract_coords_from_path(path):
    """
    Extrai coordenadas dos comandos de caminho SVG.
    O caminho segue um padrão que começa com 'M' seguido por várias 'L'.
    """
    # Regex para encontrar todos os 'L' seguidos de números (coordenadas)
    coords = re.findall(r'L(\d+\.?\d*),(\d+\.?\d*)', path)
    # Convertendo strings para float e então para int (caso tenha decimais)
    return [[int(float(x)), int(float(y))] for x, y in coords]

def convert_shapes_to_idisf_format(drawn_shapes_data):
    coords = []
    size_scribbles = []

    # Itera sobre as duas visualizações (top e side)
    for view_data in drawn_shapes_data:
        for shape_list in view_data:
            if shape_list:  # Se a lista de shapes não estiver vazia
                for shape in shape_list:
                    # Extrai coordenadas do 'path' se o tipo for 'path'
                    if shape.get('type') == 'path':
                        shape_coords = extract_coords_from_path(shape['path'])
                        coords.extend(shape_coords)
                        size_scribbles.append(len(shape_coords))

    coords = np.array(coords, dtype=np.int32)
    size_scribbles = np.array(size_scribbles, dtype=np.int32)

    return coords, size_scribbles

def convert_gray_to_rgb(img):
    """
    Converte uma imagem em escala de cinza para RGB replicando os valores em todos os três canais.
    
    Parâmetros:
    - img: Imagem de entrada, esperada em escala de cinza.
    
    Retorna:
    - Imagem convertida para RGB.
    """
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        # Replica os valores da escala de cinza nos 3 canais para RGB
        img_rgb = np.stack([img] * 3, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # A imagem já é RGB
        img_rgb = img
    else:
        raise ValueError("Formato de imagem não suportado para conversão.")
    
    return img_rgb


def base64_to_image_array(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    image_array = np.array(image)
    return image_array

def extract_image_from_figure(figure_dict):
    # Assumindo que 'figure_dict' é um dos dicts (image_display_top_figure ou image_display_side_figure)
    if figure_dict['layout']['images']:  # Checa se existe alguma imagem
        base64_string = figure_dict['layout']['images'][0]['source']  # Pegando a primeira imagem
        image_array = base64_to_image_array(base64_string)
        return image_array
    else:
        raise ValueError("Nenhuma imagem encontrada no dicionário fornecido.")

def shapes_to_segs(drawn_shapes_data, image_display_top_figure, image_display_side_figure):
    # Parâmetros do iDISF
    n0, iterations, f, c1, c2, num_obj, segm_method, all_borders = 500, 6, 4, 0.7, 0.8, 1, 1, 1
    
    # Convertendo drawn_shapes_data para coords e size_scribbles
    coords, size_scribbles = convert_shapes_to_idisf_format(drawn_shapes_data)
    
    # Convertendo imagens para arrays NumPy
    img_top = extract_image_from_figure(image_display_top_figure)
    img_side = extract_image_from_figure(image_display_side_figure)
    
    # Processamento da segmentação com iDISF_scribbles
    label_img_top, border_img_top = iDISF_scribbles(img_top, n0, iterations, coords, size_scribbles, num_obj, f, c1, c2, segm_method, all_borders)
    # label_img_side, border_img_side = iDISF_scribbles(img_side, n0, iterations, coords, size_scribbles, num_obj, f, c1, c2, segm_method, all_borders)

    # plt.imshow(label_img_top, cmap='gray')
    # plt.colorbar()
    # plt.title("Segmentation Output")
    # plt.show()

    # A função label_to_colors deve receber um numpy array, não uma URL ou string
    label_img_top_rgb = image_utils.label_to_colors(label_img_top, colormap=["#000000", "#8A2BE2"], alpha=[0, 128], color_class_offset=0)
    # label_img_side_rgb = image_utils.label_to_colors(label_img_side, colormap=["#000000", "#8A2BE2"], alpha=[0, 128], color_class_offset=0)

    # Convertendo os arrays RGB para URLs de dados apenas após a coloração
    label_img_top_url = array_to_data_url(img_as_ubyte(label_img_top_rgb))
    # label_img_side_url = array_to_data_url(img_as_ubyte(label_img_side_rgb))

    print(f"Type of label_img_top: {type(label_img_top)}, Shape: {getattr(label_img_top, 'shape', 'No shape attribute')}")


    return label_img_top_rgb


# Outputs: A callback atualiza dois outputs: "found-segs", que provavelmente contém os dados das segmentações encontradas transformadas para exibição, e "current-render-id", que é uma identificação para rastrear a renderização atual.
# Inputs: O gatilho para essa callback é a mudança nos dados de "drawn-shapes".
# States: A callback usa o estado das figuras dos gráficos de topo e lateral e o id de renderização atual como estado.
@app.callback(
    [Output("found-segs", "data"), Output("current-render-id", "data")],
    [Input("drawn-shapes", "data")],
    [
        State("image-display-graph-top", "figure"),
        State("image-display-graph-side", "figure"),
        State("current-render-id", "data"),
    ],
)
def draw_shapes_react(
    drawn_shapes_data,
    image_display_top_figure,
    image_display_side_figure,
    current_render_id,
):
    if any(
        [
            e is None
            for e in [
                drawn_shapes_data,
                image_display_top_figure,
                image_display_side_figure,
            ]
        ]
    ):
        return dash.no_update
    t1 = time.time()
    found_segs_array = shapes_to_segs(
        drawn_shapes_data, image_display_top_figure, image_display_side_figure,
    )  # This should be a numpy array now
    t2 = time.time()
    PRINT("Time to convert shapes to segments:", t2 - t1)
    # convert to a colored image
    # Supondo que `found_segs_tensor` é um numpy array com valores inteiros representando etiquetas
    fst_colored = image_utils.label_to_colors(
        found_segs_array,
        colormap=["#000000", "#8A2BE2"],  # Preto para fundo, Violeta para a etiqueta
        alpha=[0, 128],  # Fundo transparente, etiquetas semi-transparentes
        color_class_offset=0,
        labels_contiguous=True,
        no_map_zero=True
    )

    t3 = time.time()
    PRINT("Time to convert from labels to colored image:", t3 - t2)
    # fstc_slices = [
    #     [
    #         array_to_data_url(s) if np.any(s != 0) else blank_seg_slices[j]
    #         for s in np.moveaxis(fst_colored, 0, j)
    #     ]
    #     for j in range(NUM_DIMS_DISPLAYED)
    # ]
    fstc_slices = [array_to_data_url(img_as_ubyte(slice)) for slice in fst_colored]

    t4 = time.time()
    PRINT("Time to convert to data URLs:", t4 - t3)
    PRINT("Total time to compute 2D annotations:", t4 - t1)
    
    return fstc_slices, current_render_id + 1

# Em resumo, essa função é uma ferramenta útil para manipular dados codificados em 
# Base64, convertendo-os de volta para a forma binária após terem sido transmitidos
# ou armazenados como strings de texto.
def _decode_b64_slice(s):
    return base64.b64decode(s.encode())


# A função slice_image_list_to_ndarray(fstc_slices) converte uma lista de imagens 
# codificadas em Base64 (normalmente armazenadas como URLs de dados de imagens PNG) 
# em um array NumPy multidimensional.
def slice_image_list_to_ndarray(fstc_slices):
    # convert encoded slices to array
    # TODO eventually make it format agnostic, right now we just assume png and
    # strip off length equal to uri_header from the uri string
    uri_header = "data:image/png;base64,"
    # preallocating the final tensor by reading the first image makes converting
    # much faster (because all the images have the same dimensions)
    n_slices = len(fstc_slices)
    first_img = plot_common.str_to_img_ndarrary(
        _decode_b64_slice(fstc_slices[0][len(uri_header) :])
    )
    fstc_ndarray = np.zeros((n_slices,) + first_img.shape, dtype=first_img.dtype)
    PRINT("first_img.dtype", first_img.dtype)
    fstc_ndarray[0] = first_img
    for n, img_slice in enumerate(fstc_slices[1:]):
        img = plot_common.str_to_img_ndarrary(
            _decode_b64_slice(img_slice[len(uri_header) :])
        )
        fstc_ndarray[n] = img
    PRINT("fstc_ndarray.shape", fstc_ndarray.shape)
    # transpose back to original
    if len(fstc_ndarray.shape) == 3:
        # Brain data is lacking the 4th channel dimension
        # Here we allow for this function to also return an array for the 3D brain data
        return fstc_ndarray.transpose((1, 2, 0))
    return fstc_ndarray.transpose((1, 2, 0, 3))



# A função save_found_slices(fstc_slices) é projetada para converter uma lista de 
# fatias de imagem (possivelmente anotações ou segmentações de imagens médicas) em 
# um arquivo no formato NIfTI (um formato comum para armazenamento de dados de 
# imagem médica), e então codificar esse arquivo em Base64 para que possa ser 
# facilmente baixado através de uma interface web.
# Converts found slices to nii file and encodes in b64 so it can be downloaded
def save_found_slices(fstc_slices):
    # we just save the first view (it makes no difference in the end)
    fstc_slices = fstc_slices[0]
    fstc_ndarray = slice_image_list_to_ndarray(fstc_slices)
    # if the tensor is all zero (no partitions found) return None
    if np.all(fstc_ndarray == 0):
        return None
    # TODO add affine
    # technique for writing nii to bytes from here:
    # https://gist.github.com/arokem/423d915e157b659d37f4aded2747d2b3
    fstc_nii = nib.Nifti1Image(skimage.img_as_ubyte(fstc_ndarray), affine=None)
    fstcbytes = io.BytesIO()
    file_map = fstc_nii.make_file_map({"image": fstcbytes, "header": fstcbytes})
    fstc_nii.to_file_map(file_map)
    fstcb64 = base64.b64encode(fstcbytes.getvalue()).decode()
    return fstcb64


# O trecho de código que você forneceu define uma callback no aplicativo Dash que 
# lida com a funcionalidade de download para duas situações distintas, dependendo 
# de qual botão de download foi pressionado pelo usuário.
@app.callback(
    Output("found-image-tensor-data", "data"),
    [Input("download-button", "n_clicks"), Input("download-brain-button", "n_clicks")],
    [State("found-segs", "data"), State("image-slices", "data")],
)
def download_button_react(
    download_button_n_clicks,
    download_brain_button_n_clicks,
    found_segs_data,
    brain_data,
):
    ctx = dash.callback_context
    # Find out which download button was triggered
    if not ctx.triggered:
        # Nothing has happened yet
        return ""
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "download-button":
        ret = save_found_slices(found_segs_data)
    elif trigger_id == "download-brain-button":
        ret = save_found_slices(brain_data)
    else:
        return ""

    if ret is None:
        return ""
    return ret

# A função JavaScript especificada dentro da callback é responsável por converter 
# uma string codificada em Base64 em um arquivo binário (Blob) e criar uma URL 
# temporária para esse arquivo, que pode ser usada para downloads. 
# Talvez tenha algum problema com essa funcao
app.clientside_callback(
    """
function (found_image_tensor_data) {
    if (found_image_tensor_data.length <= 0) {
        return "";
    }
    // for some reason you can't use the conversion to ascii from base64 directly
    // with blob, you have to use the ascii encoded as numbers
    const byte_chars = window.atob(found_image_tensor_data);
    const byte_numbers = Array.from(byte_chars,(b,i)=>byte_chars.charCodeAt(i));
    const byte_array = new Uint8Array(byte_numbers);
    let b = new Blob([byte_array],{type: 'application/octet-stream'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download-link", "href"),
    [Input("found-image-tensor-data", "data")],
)

# Esta função garante que o arquivo seja baixado automaticamente sem que o 
# usuário precise clicar explicitamente no link
app.clientside_callback(
    """
function (href) {
    if (href != "") {
        let download_a=document.getElementById("download-link");
        download_a.click();
    }
    return '';
}
""",
    Output("dummy", "children"),
    [Input("download-link", "href")],
)

# callback executada no lado do cliente (navegador) que manipula a interação com 
# um botão que alterna entre visualizações 2D e 3D em uma interface de usuário web
app.clientside_callback(
    """
function (view_select_button_nclicks,current_render_id) {
    console.log("view_select_button_nclicks");
    console.log(view_select_button_nclicks);
    var graphs_2d = document.getElementById("2D-graphs"),
        graphs_3d = document.getElementById("3D-graphs"),
        ret = "";
    // update view select button
    var view_select_button = document.getElementById("view-select-button");
    if (view_select_button) {
        view_select_button.textContent = view_select_button_nclicks % 2 ?
            "2D View" :
            "3D View";
    }
    if (graphs_2d && graphs_3d) {
        if (view_select_button_nclicks % 2) {
            graphs_2d.style.display = "none";
            graphs_3d.style.display = "";
            ret = "3d shown";
        } else {
            graphs_2d.style.display = "grid";
            graphs_3d.style.display = "none";
            ret = "2d shown";
        }
    }
    ret += ","+current_render_id;
    return ret;
}
""",
    Output("dummy2", "children"),
    [Input("view-select-button", "n_clicks")],
    [State("current-render-id", "data")],
)

# A função store_scene_data dentro de uma callback do Dash lida com a atualização 
# e armazenamento das configurações de uma cena 3D baseada em interações do usuário, 
# como rotação, zoom, ou reposicionamento da vista em um gráfico 3D. 
# Esta função é projetada para capturar mudanças na configuração da cena 3D e 
# armazená-las, permitindo que o estado visual atual do gráfico seja preservado e 
# restaurado
@app.callback(
    Output("fig-3d-scene", "data"),
    [Input("image-display-graph-3d", "relayoutData")],
    [State("fig-3d-scene", "data")],
)
def store_scene_data(graph_3d_relayoutData, last_3d_scene):
    PRINT("graph_3d_relayoutData", graph_3d_relayoutData)
    if graph_3d_relayoutData is not None:
        for k in graph_3d_relayoutData.keys():
            last_3d_scene[k] = graph_3d_relayoutData[k]
        return last_3d_scene
    return dash.no_update


# O código que você compartilhou define uma função de callback em um aplicativo Dash 
# que é responsável por atualizar a visualização de um gráfico 3D com base em 
# diferentes entradas e estados. A função populate_3d_graph analisa e processa 
# dados de entrada, como formas desenhadas e segmentações, e renderiza esses dados 
# em uma visualização 3D. 
@app.callback(
    [Output("image-display-graph-3d", "figure"), Output("last-render-id", "data")],
    [Input("dummy2", "children"), Input("show-hide-seg-3d", "children")],
    [
        State("drawn-shapes", "data"),
        State("fig-3d-scene", "data"),
        State("last-render-id", "data"),
        State("image-display-graph-top", "figure"),
        State("image-display-graph-side", "figure"),
    ],
)
def populate_3d_graph(
    dummy2_children,
    show_hide_seg_3d,
    drawn_shapes_data,
    last_3d_scene,
    last_render_id,
    image_display_top_figure,
    image_display_side_figure,
):
    # extract which graph shown and the current render id
    graph_shown, current_render_id = dummy2_children.split(",")
    current_render_id = int(current_render_id)
    start_time = time.time()
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    # check that we're not toggling the display of the 3D annotation
    if cbcontext != "show-hide-seg-3d.children":
        PRINT(
            "might render 3D, current_id: %d, last_id: %d"
            % (current_render_id, last_render_id)
        )
        if graph_shown != "3d shown" or current_render_id == last_render_id:
            if current_render_id == last_render_id:
                PRINT("not rendering 3D because it is up to date")
            return dash.no_update
    PRINT("rendering 3D")
    segs_ndarray = shapes_to_segs(
        drawn_shapes_data, image_display_top_figure, image_display_side_figure,
    ).transpose((1, 2, 0))
    # image, color
    images = [
        (img.transpose((1, 2, 0))[:, :, ::-1], "grey"),
    ]
    if show_hide_seg_3d == "show":
        images.append((segs_ndarray[:, :, ::-1], "purple"))
    data = []
    for im, color in images:
        im = image_utils.combine_last_dim(im)
        try:
            verts, faces, normals, values = measure.marching_cubes(im, 0, step_size=3)
            x, y, z = verts.T
            i, j, k = faces.T
            data.append(
                go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.5, i=i, j=j, k=k)
            )
        except RuntimeError:
            continue
    fig = go.Figure(data=data)
    fig.update_layout(**last_3d_scene)
    end_time = time.time()
    PRINT("serverside 3D generation took: %f seconds" % (end_time - start_time,))
    return (fig, current_render_id)


# ======= Callback for modal popup =======
# A função update_click_output definida dentro de um callback no aplicativo Dash 
# manipula a visibilidade de um elemento Markdown no front-end, baseado nos cliques 
# de dois botões distintos: um para aprender mais (learn-more-button) e outro para 
# fechar (markdown_close)
@app.callback(
    Output("markdown", "style"),
    [Input("learn-more-button", "n_clicks"), Input("markdown_close", "n_clicks")],
)
def update_click_output(button_click, close_click):
    if button_click > close_click:
        return {"display": "block"}
    else:
        return {"display": "none"}
    



#funcao principal que roda o servidor
if __name__ == "__main__":
    app.run_server(debug=DEBUG)
