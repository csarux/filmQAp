# ipyfilmqa
"""
Módulo para procesar películas radiocrómicas en cuadernos python

Funcionalidades:

Subir archivos de imagen al servidor jupyter
Corregir artefactos de digitalizacion
Segmentar la imagen
Procesar la imagen y calcular la dosis
Subir los archivos de dosis calculados en el planificador
Orientar la dosis medida respecto a la dosis calculada
Exportar las imágenes de dosis en formato Varian dxf

"""
# Importación de módulos
from skimage import img_as_ubyte, img_as_float64, img_as_uint
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.util import invert
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import Button, VBox, HBox, Image
from jupyter_bbox_widget import BBoxWidget
from base64 import b64encode
from pathlib import Path
from asyncio import Future, ensure_future
import pyfilmqa as fqa
from  pyfilmqa import config, cdf, caldf
from tifffile import TiffFile
from tifffile import imread as timread, imwrite as timwrite
from PIL.Image import fromarray
import io


from pyfilmqa import config

# Definición de la clase para mantener el estado del modulo
class cFQA:
    def __init__(self, imfilename='', dcmfilename='', bbfilename='bb.csv'):
        self.imfilename = imfilename
        self.impath = Path(imfilename)
        self.dcmfilename = dcmfilename
        self.dcmpath = Path(dcmfilename)
        self.bbfilename = bbfilename
        self.bbpath = Path(bbfilename)
        self.config = config
        self.imdata = np.array([])
        self.cimdata = np.array([])
        self.Dim = np.array([])
        self.oDim = np.array([])
    def setimfile(self, filename):
        self.imfilename=filename
        self.impath=Path(filename)
    def setimpngpath(self):
        ofilepath = Path(self.imfilename)
        self.impngpath = ofilepath.with_suffix('.png')
    def setdcmfile(self, filename):
        self.dcmfilename=filename
        self.dcmpath=Path(filename)
    def setbbfile(self, filename):
        self.bbfilename=filename
        self.bbpath=Path(filename)
    def setimdata(self, imarray):
        self.imdata = imarray
    def setcimdata(self, imarray):
        self.cimdata = imarray
    def setDim(self, imarray):
        self.Dim = imarray
    def setoDim(self, imarray):
        self.oDim = imarray

# La variable de estado
st = cFQA()

# Objetos gráficos
uploader = widgets.FileUpload(accept='.tif, .tiff', multiple=False)
processor = widgets.FileUpload(accept='.dcm, .DCM', multiple=False)
outimg, outdcm = widgets.Output(), widgets.Output()
bbwidget = BBoxWidget()

def leerConfiguracion(configfile='filmQAp.config'):
    st.config.read(configfile)

def testconfig():
    print(st.config['DosePlane']['Dmax'])

def bbsave():
    bbdf = pd.DataFrame(bbwidget.bboxes)
    bbdf.to_csv(st.bbpath)
    fqa.segRegs(imfile=st.imfilename, bbfile=st.bbfilename)

bbwidget.on_submit(bbsave)

def wait_for_change(widget, value):
    future = Future()
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    return future

def tiff2png():
        aim = imread(st.impath)
        st.setimpngpath()
        imsave(st.impngpath, img_as_ubyte(rescale_intensity(aim, out_range='uint8')))

async def uploaderf():
    uploaderValue = await wait_for_change(uploader, 'value')
    filename = list(uploaderValue)[0]
    st.setimfile(filename)
    with open(filename, 'wb') as output_file:
        content = uploaderValue[filename]['content']
        output_file.write(content)
        outimg.append_stdout('Archivo subido: ' + filename + '\n')
    if st.impath.suffix == '.tif':
        tiff2png()

async def dcmprocessorf():
    processorValue = await wait_for_change(processor, 'value')
    filename = list(processorValue)[0]
    st.setdcmfile(filename)
    with open(filename, 'wb') as output_file:
        content = processorValue[filename]['content']
        output_file.write(content)
        outdcm.append_stdout('Archivo subido: ' + filename + '\n')
    if st.dcmpath.suffix == '.dcm':
        dcm2dxf()

def subirImagenDigitalizada():
    ensure_future(uploaderf())
    display(uploader, outimg)

def exportarPlanoDosisPlanificador():
    ensure_future(dcmprocessorf())
    display(processor, outdcm)

def corregirScanner():
    pixels = widgets.Text(
        value='',
        placeholder='Secuencia ej: 150, 145',
        description='Pixels:',
        disabled=False
    )
    factors = widgets.Text(
        value='',
        placeholder='Secuencia ej: 0.91, 0.95',
        description='Factores:',
        disabled=False
    )

    correct = widgets.Button(
        description='Corregir'
    )

    save = widgets.Button(
        description='Actualizar'
    )

    restore = widgets.Button(
        description='Restaurar'
    )

    im = imread(st.imfilename)
    st.setimdata(im)
    h, w, ch = im.shape

    fig, (axo, axc) = plt.subplots(ncols=2)
    imub = img_as_ubyte(rescale_intensity(im, out_range='uint8'))
    imcub = img_as_ubyte(rescale_intensity(im, out_range='uint8'))
    axo.set_title('Imagen original')
    axc.set_title('Imagen corregida')
    axo.imshow(imub)
    axc.imshow(imcub)

    def imcorr(im, ps, fs):
        """
        Corrects the scan calibration artifacts
        """
        imf=img_as_float64(im)
        imfo=img_as_float64(im)

        for p, f in zip(ps, fs):
            imf[:, p, :] = imf[:, p, :] * f

        imf[imfo[...]>0.9]=imfo[imfo[...]>0.9]
        imf[imf[...]>1]=1
        imc = img_as_uint(imf)
        st.setcimdata(imc)

    def on_clicked_correct(b):
        ps = list(map(int, pixels.value.split(',')))
        fs = list(map(float, factors.value.split(',')))
        im = imread(st.imfilename)
        imcorr(im, ps, fs)
        imcub = img_as_ubyte(rescale_intensity(st.cimdata, out_range='uint8'))
        axc.imshow(imcub)

    def on_clicked_save(b):
        fqa.imgUpdate(imfile=st.imfilename, imdata=st.cimdata)
        tiff2png()

    def on_clicked_restore(b):
        fqa.imgUpdate(imfile=st.imfilename, imdata=st.imdata)
        tiff2png()
        imcub = img_as_ubyte(rescale_intensity(st.imdata, out_range='uint8'))
        axc.imshow(imcub)

    correct.on_click(on_clicked_correct)

    save.on_click(on_clicked_save)

    restore.on_click(on_clicked_restore)

    display(widgets.VBox([pixels, factors, widgets.HBox([correct, save, restore])]))

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(b64encode(image_bytes), 'utf-8')
    return "data:image/png;base64,"+encoded

def segmentarImagen():
    bbwidget.image=encode_image(st.impngpath.name)
    bbwidget.classes=['Film', 'Calibration', 'Background', 'Center']
    display(bbwidget)


def representarPlanoDosisPlanificador():
    print('Plano de dosis calculado:')
    pDim = fqa.DICOMDose(dcmfile=st.dcmfilename)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(pDim, cmap=plt.cm.gray_r)
    plt.show()

def representarOrientacionesPlanoDosisPelicula():
    cDim = st.Dim
    fig, ((ax1, ax2, ax3, ax4), (axr1, axr2, axr3, axr4)) = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    ax1.set_axis_off()
    ax1.imshow(cDim, cmap=plt.cm.gray_r)
    ax2.set_axis_off()
    ax2.imshow(np.rot90(cDim, 1), cmap=plt.cm.gray_r)
    ax3.set_axis_off()
    ax3.imshow(np.rot90(cDim, 2), cmap=plt.cm.gray_r)
    ax4.set_axis_off()
    ax4.imshow(np.rot90(cDim, 3), cmap=plt.cm.gray_r)
    fDim = np.fliplr(cDim)
    axr1.set_axis_off()
    axr1.imshow(fDim, cmap=plt.cm.gray_r)
    axr2.set_axis_off()
    axr2.imshow(np.rot90(fDim, 1), cmap=plt.cm.gray_r)
    axr3.set_axis_off()
    axr3.imshow(np.rot90(fDim, 2), cmap=plt.cm.gray_r)
    axr4.set_axis_off()
    axr4.imshow(np.rot90(fDim, 3), cmap=plt.cm.gray_r)
    plt.show()

def on_b_clicked(b):
    st.setoDim(st.Dim)
    exportarPlanoDosisPelicula()

def on_b90_clicked(b90):
    st.setoDim(np.rot90(st.Dim))
    exportarPlanoDosisPelicula()

def on_b180_clicked(b180):
    st.setoDim(np.rot90(st.Dim, k=2))
    exportarPlanoDosisPelicula()

def on_b270_clicked(b270):
    st.setoDim(np.rot90(st.Dim, k=3))
    exportarPlanoDosisPelicula()

def on_bf_clicked(bf):
    st.setoDim(np.fliplr(st.Dim))
    exportarPlanoDosisPelicula()

def on_bf90_clicked(bf90):
    st.setoDim(np.rot90(np.fliplr(st.Dim)))
    exportarPlanoDosisPelicula()

def on_bf180_clicked(bf180):
    st.setoDim(np.rot90(np.fliplr(st.Dim)), k=2)
    exportarPlanoDosisPelicula()

def on_bf270_clicked(bf270):
    st.setoDim(np.rot90(np.fliplr(st.Dim)), k=3)
    exportarPlanoDosisPelicula()

def compress_to_bytes(data, fmt):
    """
    Helper function to compress image data via PIL/Pillow.
    """
    buff = io.BytesIO()
    img = fromarray(data)
    img.save(buff, format=fmt)

    return buff.getvalue()

def createImageWidget(im):
    imdata = compress_to_bytes(img_as_uint(im),'png')

    iw = Image(
        value=imdata,
        format='png',
        width=im.shape[1],
        height=im.shape[0]
    )

    return iw

def reorientarYExportarPlanoDosisPelicula():
    representarPlanoDosisPlanificador()

    print('Seleccionar la orientación correcta del plano de dosis medido:')

    words = ['Dejar igual', 'Rotar 90', 'Rotar 180', 'Rotar 270', 'Voltear', 'Voltear y rotar 90', 'Voltear y rotar 180', 'Voltear y rotar 270']
    items = [Button(description=w) for w in words]

    [b, b90, b180, b270] = [items[0], items[1], items[2], items[3]]
    [bf, bf90, bf180, bf270] = [items[4], items[5], items[6], items[7]]

    Dmax = float(config['DosePlane']['Dmax'])
    pDim = fqa.DICOMDose(dcmfile=st.dcmfilename)
    pDmax = pDim.max()

    imDn = invert(st.Dim / pDmax)

    [iw, iw90, iw180, iw270] = [
        createImageWidget(imDn),
        createImageWidget(np.rot90(imDn)),
        createImageWidget(np.rot90(imDn, k=2)),
        createImageWidget(np.rot90(imDn, k=3)),
    ]

    for i in [iw, iw90, iw180, iw270]:
        i.layout.object_fit = 'contain'

    [iwf, iwf90, iwf180, iwf270] = [
        createImageWidget(np.fliplr(imDn)),
        createImageWidget(np.rot90(np.fliplr(imDn))),
        createImageWidget(np.rot90(np.fliplr(imDn), k=2)),
        createImageWidget(np.rot90(np.fliplr(imDn), k=3)),
    ]

    for i in [iwf, iwf90, iwf180, iwf270]:
        i.layout.object_fit = 'scale-down'

    b.on_click(on_b_clicked)
    b90.on_click(on_b90_clicked)
    b180.on_click(on_b180_clicked)
    b270.on_click(on_b270_clicked)
    bf.on_click(on_bf_clicked)
    bf90.on_click(on_bf90_clicked)
    bf180.on_click(on_bf180_clicked)
    bf270.on_click(on_bf270_clicked)

    vb = VBox([b, iw])
    vb90 = VBox([b90, iw90])
    vb180 = VBox([b180, iw180])
    vb270 = VBox([b270, iw270])

    vbf = VBox([bf, iwf])
    vbf90 = VBox([bf90, iwf90])
    vbf180 = VBox([bf180, iwf180])
    vbf270 = VBox([bf270, iwf270])

    upperhbox = HBox([vb, vb90, vb180, vb270])
    lowerhbox = HBox([vbf, vbf90, vbf180, vbf270])

    display(VBox([upperhbox, lowerhbox]))

def reorientarYExportarPlanoDosisPelicula_original():
    representarPlanoDosisPlanificador()

    im = st.Dim

    print('Plano de dosis medido:')

    fig, ((ax, ax90, ax180, ax270), (axf, axf90, axf180, axf270)) = plt.subplots(ncols=4, nrows=2, figsize=(8,4))
    ax.imshow(im, cmap=plt.cm.gray_r)
    im90 = np.rot90(im)
    ax90.imshow(im90, cmap=plt.cm.gray_r)
    im180 = np.rot90(im, k=2)
    ax180.imshow(im180, cmap=plt.cm.gray_r)
    im270 = np.rot90(im, k=3)
    ax270.imshow(im270, cmap=plt.cm.gray_r)

    imf = np.fliplr(im)
    axf.imshow(imf, cmap=plt.cm.gray_r)
    imf90 = np.rot90(np.fliplr(im))
    axf90.imshow(imf90, cmap=plt.cm.gray_r)
    imf180 = np.rot90(np.fliplr(im), k=2)
    axf180.imshow(imf180, cmap=plt.cm.gray_r)
    imf270 = np.rot90(np.fliplr(im), k=3)
    axf270.imshow(imf270, cmap=plt.cm.gray_r)

    ax.set_axis_off()
    ax.text(50, 50, 'Dejar igual', color='r')
    ax90.set_axis_off()
    ax90.text(50, 50, 'Rotar 90', color='r')
    ax180.set_axis_off()
    ax180.text(50, 50, 'Rotar 180', color='r')
    ax270.set_axis_off()
    ax270.text(50, 50, 'Rotar 270', color='r')
    axf.set_axis_off()
    axf.text(50, 50, 'Voltear', color='r')
    axf90.set_axis_off()
    axf90.text(50, 50, 'Voltear y rotar 90', color='r')
    axf180.set_axis_off()
    axf180.text(50, 50, 'Voltear y rotar 180', color='r')
    axf270.set_axis_off()
    axf270.text(50, 50, 'Voltear y rotar 270', color='r')

    plt.tight_layout()

    words = ['Dejar igual', 'Rotar 90', 'Rotar 180', 'Rotar 270', 'Voltear', 'Voltear y rotar 90', 'Voltear y rotar 180', 'Voltear y rotar 270']
    items = [Button(description=w) for w in words]
    [b, b90, b180, b270] = [items[0], items[1], items[2], items[3]]
    [bf, bf90, bf180, bf270] = [items[4], items[5], items[6], items[7]]
    upperbs = [b, b90, b180, b270]
    lowerbs = [bf, bf90, bf180, bf270]
    upper_box = HBox(upperbs)
    lower_box = HBox(lowerbs)

    b.on_click(on_b_clicked)
    b90.on_click(on_b90_clicked)
    b180.on_click(on_b180_clicked)
    b270.on_click(on_b270_clicked)
    bf.on_click(on_bf_clicked)
    bf90.on_click(on_bf90_clicked)
    bf180.on_click(on_bf180_clicked)
    bf270.on_click(on_bf270_clicked)


    display(VBox([upper_box, lower_box]))


def dcm2dxf():
    print('Convertir a formato dxf el plano de dosis calculado en Eclipse...')
    demodict = fqa.DICOMDemographics(dcmfile=st.dcmfilename)
    pxsp = fqa.DICOMPixelSpacing(dcmfile=st.dcmfilename)
    imsz = fqa.DICOMImageSize(dcmfile=st.dcmfilename)
    pDim = fqa.DICOMDose(dcmfile=st.dcmfilename)
    dxffilePath = Path('/home/radiofisica/Shares/Radiofisica/Medidas Pacientes/IMRT/' + demodict['PatientId1'] + '/Plan.dxf')
    fqa.dxfWriter(Data=pDim, dxfFileName=dxffilePath, DataOrigin=st.dcmfilename,
                  AcqType='Predicted Portal', PatientId1=demodict['PatientId1'],
                  PatientId2=demodict['PatientId1'], LastName=demodict['LastName'],
                  FirstName=demodict['FirstName'], pxsp=pxsp, imsz=imsz)
    outdcm.append_stdout('Exportado archivo Radiofisica/Medidas Pacientes/IMRT/'  + demodict['PatientId1'] + '/Plan.dxf')
    print('Exportado archivo Radiofisica/Medidas Pacientes/IMRT/'  + demodict['PatientId1'] + '/Plan.dxf')

def tif2dxf():
    print('Convertir a formato dxf el plano de dosis medido con película...')
    demodict = fqa.DICOMDemographics(dcmfile=st.dcmfilename)
    pxsp = fqa.TIFFPixelSpacing(st.imfilename)
    imsz = fqa.DoseImageSize(st.oDim)
    dxffilePath = Path('/home/radiofisica/Shares/Radiofisica/Medidas Pacientes/IMRT/' + demodict['PatientId1'] + '/Film.dxf')

    fqa.dxfWriter(Data=st.oDim, dxfFileName=dxffilePath, DataOrigin=st.imfilename,
                  AcqType='Acquired Portal', PatientId1=demodict['PatientId1'],
                  PatientId2=demodict['PatientId1'], LastName=demodict['LastName'],
                  FirstName=demodict['FirstName'], pxsp=pxsp, imsz=imsz)
    print('Exportado archivo Radiofisica/Medidas Pacientes/IMRT/'  + demodict['PatientId1'] + '/Film.dxf')

def procesarPelicula():
    print('Determinación de las coordenadas para la corrección lateral...')
    cdf = fqa.coordOAC(imfile=st.imfilename)
    print('Determinación del fondo...')
    abase = fqa.baseDetermination(imfile=st.imfilename, config=st.config)
    print('Calibración de la digitalización...')
    caldf=fqa.PDDCalibration(config=st.config, imfile=st.imfilename, base=abase)
    # Determinación de la dosis en cada canal
    st.setDim(fqa.mphspcnlmprocf_multiprocessing(imfile=st.imfilename, config=st.config, caldf=caldf, ccdf=cdf))
    # Postprocesado de la imagen
    st.setDim(fqa.postmphspcnlmprocf(st.Dim, config=st.config, planfile=st.dcmfilename))

def exportarPlanoDosisPelicula():
    tif2dxf()
