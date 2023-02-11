# Module pyFilmQA
# File pyFilmQA.py
# Version 0.1

"""
A module to process radiochromic films and to export the results in dxf format.
It provides also a converter from 2D dose distributions in dicom files to dxf format.
dxf is a format used by Varian as an interchange data tool.
Dose planes in dxf can be imported from Portal Dosinmetry application.

"""

# - Config file
import configparser
# - Numerical calculations
import numpy as np
# - Data mamaging
import pandas as pd
# - File lists
import glob
# - Dates and file formats
import os
from datetime import datetime
# - Paths and file extensions managing
from pathlib import Path
# - TIFF files
from tifffile import TiffFile
from tifffile import imread as timread, imwrite as timwrite
from json import dumps
# - DICOM files editing
import pydicom as dicom
# - Image processing
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_uint
from skimage.measure import profile_line
from skimage.transform import resize
from skimage.feature import canny
from skimage.morphology import label
from skimage.measure import regionprops
# - Non-local means
from skimage.restoration import denoise_nl_means
# - Interpolation
from scipy.interpolate import interp1d
# - Numerical non linear fits
from scipy.optimize import curve_fit
from scipy.optimize import minimize
# - Non linear function inversion
from scipy.optimize import fsolve
# - Calibration models fits
from lmfit import Parameters, Model
# - Progress bar
from tqdm.notebook import tqdm
# - To pass additional parameters to map function
from itertools import repeat
# - Multiprocessing
from multiprocessing import Pool


# Module objects
config = configparser.ConfigParser()
cdf = pd.DataFrame()
abase = np.array([])
caldf = pd.DataFrame()
cDim = np.array([])

def imgUpdate(imfile=None, imdata=None, dcmfile=None):
    """
    A function to update the data of the film image
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    imdata : numpy array
        A 3D array containing the corrected image data
        
    Returns
    -------
    No value returned
        
    """
    filename = Path(imfile)

    with TiffFile(imfile) as tif:
        page_tags = tif.pages[0].tags
        # Guardar cada región un archivo 
        # Extratags
        metadata_tag = dumps({"PatientId": '001PATFILM', "PatientName": 'Medidas', 'PatientFamilyName' : 'Películas', 'Sex' : 'Male'})
        if dcmfile:
            demodict = DICOMDemographics(dcmfile=dcmfile) 
            metadata_tag = dumps({"PatientId": demodict['PatientId1'], "PatientName": demodict['FirstName'], 'PatientFamilyName' : demodict['LastName']})
        extra_tags = [("MicroManagerMetadata", 's', 0, metadata_tag, True),
                      ('Make', 's', 0, page_tags['Make'].value, True),
                      ('Model', 's', 0, page_tags['Model'].value, True),
                      ('DateTime', 's', 0, page_tags['DateTime'].value, True),              
                      ("ProcessingSoftware", 's', 0, "pyfilmqa", True)]
#                      ("MetaData", 's', 0, metadata_tag, True)]
    timwrite(filename, imdata, extratags=extra_tags)
    with TiffFile(filename, mode='r+b') as tif:
        tif.pages[0].tags['XResolution'].overwrite((720000, 10000))
        tif.pages[0].tags['YResolution'].overwrite((720000, 10000))

def segRegs(imfile=None, bbfile='bb.csv'):
    """
    A function to segment the film image
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    bbfile : str
        the name of the bounding box file, the file containing the position and size of the bounding boxes of the image file. It should be a csv file.
        
    Returns
    -------
    No value returned
        
    """
    # Leer el archivo de regiones de interes y generar un dataframe con las coordenas y dimensiones de las regiones
    bbdf = pd.read_csv(bbfile)
    # Nombre del archivo de imagen
    filename = Path(imfile)
    # Segmentar el archivo de imagen
    im = imread(imfile)
    for r in ['Film', 'Calibration', 'Background', 'Center']:
        reg = bbdf.loc[bbdf.label == r]
        rim = im[reg.y.values[0]:reg.y.values[0]+reg.height.values[0], reg.x.values[0]:reg.x.values[0]+reg.width.values[0], :]
        # Leer los metadatos del archivo de imagen
        with TiffFile(imfile) as tif:
            page_tags = tif.pages[0].tags
            # Guardar cada regiÃ³n un archivo 
            # Extratags
            metadata_tag = dumps({"PatientId": '001PATFILM', "PatientName": 'Prueba', 'PatientFamilyName' : 'PelÃ­culas', 'Sex' : 'Male'})
            extra_tags = [("MicroManagerMetadata", 's', 0, metadata_tag, True),
                          ('Make', 's', 0, page_tags['Make'].value, True),
                          ('Model', 's', 0, page_tags['Model'].value, True),
                          ('DateTime', 's', 0, page_tags['DateTime'].value, True),              
                          ("ProcessingSoftware", 's', 0, "pyFilmQAModule", True)]
            timwrite(filename.with_suffix('.' + str(r) + '.tif'), rim, extratags=extra_tags)
        
def twriteDoseFilm(imfile=None, doseim=None):
    """
    A function to export in TIFF the calculated dose image from the film data
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    doseim : numpy array 
        A numpy array containing the spatial dose distribution

    Returns
    -------
    No value returned
        
    """
    # Nombre del archivo de imagen
    filename = Path(imfile)
    # Leer los metadatos del archivo de imagen
    with TiffFile(imfile) as tif:
        page_tags = tif.pages[0].tags
    extra_tags = [('Make', 's', 0, page_tags['Make'].value, True),
                  ('Model', 's', 0, page_tags['Model'].value, True),
                  ('DateTime', 's', 0, page_tags['DateTime'].value, True),              
                  ("ProcessingSoftware", 's', 0, "pyFilmQAModule", True)]
    timwrite(filename.with_suffix('.' + 'dose' + '.tif'), doseim, extratags=extra_tags)
    

def coordOAC(imfile=None, bbfile='bb.csv'):
    """
    A function to segment the film image
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    bbfile : str
        the name of the bounding box file, the file containing the position and size of the bounding boxes of the image file. It should be a csv file.
        
    Returns
    -------
    cdf : pandas DataFrame
        A pandas DataFrame containing the relavant coordiantes for the off-axis spatial correction
        
    """
    bbdf = pd.read_csv(bbfile)
    
    creg = bbdf.loc[bbdf.label == 'Center'] # Image center
    o = creg.x.values[0] + int(creg.width.values[0]/2)
    calreg = bbdf.loc[bbdf.label == 'Calibration'] # Calibration strip
    c = calreg.x.values[0] + int(calreg.width.values[0]/2)
    dosereg = bbdf.loc[bbdf.label == 'Film'] # Film dose region
    p0 = dosereg.x.values[0]
    
    pxsp = TIFFPixelSpacing(imfile=imfile)
    s = pxsp[0]
    
    lcdf = pd.DataFrame({'o' : o, 'c' : c, 'p0' : p0, 's' : s}, index=[0])
    
    cdf = lcdf.copy(deep=True)
    
    return lcdf
    
def baseDetermination(imfile=None, config=None):
    """
    A function to calculate the base value in every color channel
    
    ...
    
    Attributes
    ---------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    
    config : ConfigParser
        An object with the functionalities of the configparser module
        
    Returns
    -------
    imbase : float64 numpy array
        An array containing the value of the base digital signal in every color channel.
        
    """
    # Derivar el nombre del archivo de fondo
    bkgfilename = Path(imfile)
    bkgfilename = bkgfilename.with_suffix('.Background.tif')
    # Leer la imagen de fondo
    fim = imread(bkgfilename)
    # Tomar el valor del margen del arhivo de configuraciÃ³n
    mrg = int(config['Base']['margin'])
    # Tomar la parte central
    fim = fim[mrg:-mrg, mrg:-mrg, :]
    # Devolver el valor del fondo en cada canal
    return np.log10(2**16/fim.mean(axis=(0,1)))
    
def scandpi(imfile=None):
    """
    A function to get the scan spatial resolution

    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
    
    Returns
    -------
    dpi = float64
        The number of dots per inch in the scanning image
    """
    with TiffFile(imfile) as tif:
        page_tags = tif.pages[0].tags
    xres = page_tags['XResolution'].value
    yres = page_tags['YResolution'].value
    xdpi = xres[0]/xres[1]
    ydpi = yres[0]/yres[1]
    if xdpi == ydpi:
        dpi = xdpi
    else:
        dpi=None
    return dpi
    
def nlmf(imfile=None, config=None):
    """
    A function to denoise a multichannel image using a non-local means procedure.

    imfile : str
        The name of the image file, the file containing the image to be denoised.
    
    config : ConfigParser
        An object with the functionalities of the configparser module

    Returns
    -------
    udim = unsigned int numpy array
        A numpy array of unsigned ints with shape (xpixels, ypixels, channels)
    """
    im = imread(imfile)
    fim=img_as_float(im)
    dim=denoise_nl_means(fim, 
                         patch_size = int(config['NonLocalMeans']['PatchSize']),
                         patch_distance = int(config['NonLocalMeans']['PatchDistance']),
                         h=float(config['NonLocalMeans']['h']), 
                         channel_axis=int(config['NonLocalMeans']['ChannelAxis']))
    udim=img_as_uint(dim)
    
    return udim
    
def calf(D, f, phir, kr, phib, kb):
    """
    The two phase polymer model dose response general function for every channel
    
    ...
    
    Attributes
    ----------
    D : float64
        Absorbed dose
        
    f : float64
        The base optical density
        
    phir : float64
        The relative abundance of the red phase polymer
    
    kr : float64
        The exponent in the saturation term of the red phase polymer
        
    phib : float64
        The relative abundance of the blue phase polymer
    
    kb : float64
        The exponent in the saturation term of the blue phase polymer
        
    Returns
    -------
    d : float64
        The optical density in each channel corresponding to the absorbed dose D following the sensitometry model based on the growth of two polymer phases
    
    """
    
    return f + phir * (1-np.exp(-kr*D)) + phib * (1-np.exp(-kb*D)) 

def iratf(d, a, b, c):
    """
    The calibration function following a sensitometric model bases in rational functions and using the optical density as variable
    
    ...
    
    Attributes
    ----------
    d : float64
        Optical density as measured by the scanner
        
    a : float64
        First rational function parameter
        
    b : float64
        Second rational function parameter
    
    c : float64
        Third rational function parameter
        
    Returns
    -------
    D : float64
        The abosrbed dose D corresponding to the optical density d in each channel following the sensitometry model based on rational functions
    
    """
    
    return (a - c * 10**-d)/(10**-d - b)
    
def iratSf(S, a, b, c, Sb):
    """
    The calibration function following a sensitometric model bases in rational functions and using the digital signal as variable
    
    ...
    
    Attributes
    ----------
    S : unsigned int 16
        The digital signal measured by the scanner in every color channel
        
    a : float64
        First rational function parameter
        
    b : float64
        Second rational function parameter
    
    c : float64
        Third rational function parameter
        
    Sb : unsigned int 16
        The base digital signal determined for the film

    Returns
    -------
    D : float64
        The abosrbed dose D corresponding to the digital signal S in each channel following the sensitometry model based on rational functions
    
    """
    
    return (a - c * S/Sb)/(S/Sb - b)
    
def deriv_iratSf(S, a, b, c, Sb):
    """
    The derivative of the calibration function following a sensitometric model bases in rational functions and using the digital signal as variable
    
    ...
    
    Attributes
    ----------
    S : unsigned int 16
        The digital signal measured by the scanner in every color channel
        
    a : float64
        First rational function parameter
        
    b : float64
        Second rational function parameter
    
    c : float64
        Third rational function parameter
        
    Sb : unsigned int 16
        The base digital signal determined for the film

    Returns
    -------
    D : float64
        The abosrbed dose D corresponding to the digital signal S in each channel following the sensitometry model based on rational functions
    
    """
    
    return (c * S/Sb * (1 - 1/Sb) + c * b/Sb - a)/(S/Sb - b)**2
    
def readCalParms(config=None):
    """
    A function to read the established standard calibration parameters for the EBT3 film measured by the Microtek 1000 XL scanner
    
    ...
    
    Attributes
    ----------
    config : ConfigParser
        An object with the functionalities of the configparser module
        
    Returns
    -------
    caldf : DataFrame
        A pandas DataFrame with the calibration parameters for every color channel (multiphase model)
    
    """

    configpath = config['DEFAULT']['configpath'] 
    modelsfile = config['Models']['File']
    modelsheet = config['Models']['mphSheet']
    
    caldf = pd.read_excel(configpath + modelsfile, sheet_name=modelsheet)
    caldf.set_index('Unnamed: 0', inplace=True)
    caldf.index.names = ['ch']
    return caldf
    
def readRatParms(config=None):
    """
    A function to read a standard calibration set of parameters following the rational model for the EBT3 film measured by the Microtek 1000 XL scanner
    
    ...
    
    Attributes
    ----------
    config : ConfigParser
        An object with the functionalities of the configparser module
        
    Returns
    -------
    ratdf : DataFrame
        A pandas DataFrame with the calibration parameters for every color channel (rational model)
    
    """

    configpath = config['DEFAULT']['configpath'] 
    modelsfile = config['Models']['File']
    modelsheet = config['Models']['racSheet']
    ratdf = pd.read_excel(configpath + modelsfile, sheet_name=modelsheet)
    ratdf.set_index('Unnamed: 0', inplace=True)
    ratdf.index.names = ['ch']
    return ratdf
    
def rootcalf(D, d, f, phir, kr, phib, kb):
    """
    An internal module use function.
    It expresses the nonlinear equation to get the absorbed dose D from the optical density d using the multiphase model
    
    ...
    

    Returns
    -------
    rootcalf : float64
        The difference between the measured optical density d and the optical density as calculated for the multiphase model for the absorbed dose D
    
    """
    
    return d - calf(D, f, phir, kr, phib, kb)

def icalf(d, Dsem, f, phir, kr, phib, kb):
    """
    The calibration function following the multiphase model
    
    ...
    
    Attributes
    ----------
    d : float64
        The measured optical density
    Dsem : float64
        A seed value of the absorbed D to solve the nonlinear equation
    f : float64
        The base optical density
        
    phir : float64
        The relative abundance of the red phase polymer
    
    kr : float64
        The exponent in the saturation term of the red phase polymer
        
    phib : float64
        The relative abundance of the blue phase polymer
    
    kb : float64
        The exponent in the saturation term of the blue phase polymer
        
    Returns
    -------
    D : float64
        The calculated absorbed dose D corresponding to the measured optical density d
    
    """

    return fsolve(rootcalf, Dsem, (d, *[f, phir,  kr, phib, kb]))[0]

def Ricalf(d, rcalps, rratps):
    """
    The calibration function following the multiphase model for the red channel
    
    ...
    
    Attributes
    ----------
    d : float64
        The measured optical density
        
    rcalps : 1D numpy array
        The current scan calibration parameters for the red channel
        
    rratps : 1D numpy array
        The current scan calibration rational approximation for the red channel

    Returns
    -------
    D : float64
        The calculated absorbed dose D corresponding to the measured optical density d for the red channel
    
    """
    return icalf(d, iratf(d, *rratps), *rcalps)

Ricalfv = np.vectorize(Ricalf, excluded={1, 2})

def Gicalf(d, gcalps, gratps):
    """
    The calibration function following the multiphase model for the green channel
    
    ...
    
    Attributes
    ----------
    d : float64
        The measured optical density
        
    gcalps : 1D numpy array
        The current scan calibration parameters for the green channel
        
    gratps : 1D numpy array
        The current scan calibration rational approximation for the green channel


    Returns
    -------
    D : float64
        The calculated absorbed dose D corresponding to the measured optical density d for the green channel
    
    """
    
    return icalf(d, iratf(d, *gratps), *gcalps)

Gicalfv = np.vectorize(Gicalf, excluded={1, 2})

def Bicalf(d, bcalps, bratps):
    """
    The calibration function following the multiphase model for the blue channel
    
    ...
    
    Attributes
    ----------
    d : float64
        The measured optical density
        
    bcalps : 1D numpy array
        The current scan calibration parameters for the blue channel
        
    bratps : 1D numpy array
        The current scan calibration rational approximation for the blue channel


    Returns
    -------
    D : float64
        The calculated absorbed dose D corresponding to the measured optical density d for the blue channel
    
    """
    
    return icalf(d, iratf(d, *bratps), *bcalps)
    
Bicalfv = np.vectorize(Bicalf, excluded={1, 2})

def PDDCalibration(config=None, imfile=None, base=None):
    """
    A function to get the current scan calibration parameters 
    
    ...
    
    Attributes
    ----------
    config : ConfigParser
        An object with the functionalities of the configparser module

    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    base : 1D numpy array
        Array containing the calculated base values for every color channel

    Returns
    -------
    caldf : pandas DataFrame
        The current scan calibration parameters
    
    """
    
    # Read the calculated calibration absorbed dose distributiom (PDD)
    pddcalibfile = config['Calibration']['File']
    cdf = pd.read_excel(pddcalibfile)
    
    # Read the calibration image segment data
    calfilename = Path(imfile)
    calfilename = calfilename.with_suffix('.Calibration.tif')
    cim = imread(calfilename)
    
    # Denoise
    dcim = nlmf(imfile=calfilename, config=config)
    
    # Calculate spatial coordinates
    with TiffFile(imfile) as tif:
        page_tags = tif.pages[0].tags
    xres = page_tags['XResolution'].value
    dpi = xres[0]/xres[1]
    zres = 2.54/dpi
    zv = np.arange(0, (cim.shape[0]+0.5)*zres, zres)

    # Depth dose distribution in digital signal units
    dch, dcw, _chanels = dcim.shape
    cdd = profile_line(dcim, src=(0, dcw/2), dst=(dch, dcw/2), linewidth=20)
    
    # Depth dose distribution in optical density units
    ddf = pd.DataFrame({'z': zv, 'dr' : np.log10(2**16/cdd[:,0]), 'dg' : np.log10(2**16/cdd[:,1]), 'db' : np.log10(2**16/cdd[:,2])})
    ddf.replace([np.inf, -np.inf], np.nan, inplace=True)
    ddf.dropna(inplace=True)
    
    # Resampling to the used ratiotherapy planning system spatial resolution
    zsh, zmin, zmax = float(config['Calibration']['shift']), float(config['Calibration']['depthmin']), float(config['Calibration']['depthmax'])
    
    # Interpolation functions
    rddf = interp1d(ddf.z + zsh, ddf.dr, bounds_error=False)
    gddf = interp1d(ddf.z + zsh, ddf.dg, bounds_error=False)
    bddf = interp1d(ddf.z + zsh, ddf.db, bounds_error=False)
    
    # Add to the calibration DataFrame the optical density corresponding to the radiotherapy planning system data points
    cdf['dr'] = rddf(cdf.z)
    cdf['dg'] = gddf(cdf.z)
    cdf['db'] = bddf(cdf.z)
    
    # Filter the calibration relevant depths
    cdf = cdf.loc[(cdf.z > zmin) & (cdf.z < zmax)]
    
    # Drop NA values
    cdf.dropna(inplace=True)
    
    # Read the standard calibration parameters (multiphse model)
    configpath = config['DEFAULT']['configpath'] 
    modelsfile = config['Models']['File']
    modelsheet = config['Models']['mphSheet']
    caldf = pd.read_excel(configpath + modelsfile, sheet_name=modelsheet)
    caldf.set_index('Unnamed: 0', inplace=True)
    caldf.index.names = ['ch']
    
    # Read the standard calibration parameters (rational model)
    modelsheet = config['Models']['racSheet']
    racdf = pd.read_excel(configpath + modelsfile, sheet_name=modelsheet)
    racdf.set_index('Unnamed: 0', inplace=True)
    racdf.index.names = ['ch']
    
    # Extract every color channel parameters
    # Multiphase
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values
    # Rational
    rracps = racdf.iloc[0].values
    gracps = racdf.iloc[1].values
    bracps = racdf.iloc[2].values
    
    # Calibration models for every channel
    rcalfmodel = Model(calf)
    gcalfmodel = Model(calf)
    bcalfmodel = Model(calf)
    
    # Parameter initialization
    # Red
    rcalfparams = rcalfmodel.make_params(
        f = base[0],
        phir = rcalps[1],
        kr = rcalps[2],
        phib = rcalps[3],
        kb = rcalps[4]
    )
    
    rcalfparams['f'].vary = False
    rcalfparams['phir'].min = 0
    rcalfparams['kr'].vary = False
    rcalfparams['phib'].min = 0
    rcalfparams['kb'].vary = False
    
    # Green
    gcalfparams = gcalfmodel.make_params(
        f = base[1],
        phir = gcalps[1],
        kr = gcalps[2],
        phib = gcalps[3],
        kb = gcalps[4],
    )
    
    gcalfparams['f'].vary = False
    gcalfparams['phir'].min = 0
    gcalfparams['kr'].vary = False
    gcalfparams['phib'].min = 0
    gcalfparams['kb'].vary = False
    
    # Blue
    bcalfparams = bcalfmodel.make_params(
        f = base[2],
        phir = bcalps[1],
        kr = bcalps[2],
        phib = bcalps[3],
        kb = gcalps[4],
    )
    
    bcalfparams['f'].vary = False
    bcalfparams['phir'].min = 0
    bcalfparams['kr'].vary = False
    bcalfparams['phib'].min = 0
    bcalfparams['kb'].vary = False
    
    # Fit
    rcalfresult = rcalfmodel.fit(cdf.dr, rcalfparams, D = cdf.D)
    gcalfresult = gcalfmodel.fit(cdf.dg, gcalfparams, D = cdf.D)
    bcalfresult = bcalfmodel.fit(cdf.db, bcalfparams, D = cdf.D)    
    
    # Parameter reorganization
    caldf = pd.DataFrame({'f' : base[0], 'phir' : rcalfresult.params.get('phir').value, 'kr' : rcalps[2], 'phib' : rcalfresult.params.get('phib').value, 'kb' : rcalps[4]}, index=['R'])
    tcaldf = pd.DataFrame({'f' : base[1], 'phir' : gcalfresult.params.get('phir').value, 'kr' : gcalps[2], 'phib' : gcalfresult.params.get('phib').value, 'kb' : gcalps[4]}, index=['G'])
    caldf = pd.concat([caldf, tcaldf])
    tcaldf = pd.DataFrame({'f' : base[2], 'phir' : bcalfresult.params.get('phir').value, 'kr' : bcalps[2], 'phib' : bcalfresult.params.get('phib').value, 'kb' : bcalps[4]}, index=['B'])
    caldf = pd.concat([caldf, tcaldf])
    
    # Return the current scan calibration parameter DataFrmme
    return caldf
    
def mphnlmprocf(imfile=None, config=None, caldf=None):
    """
    A function to process the dose distribution image using nonlocal means denoising and the multiphase calibration model
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    config : ConfigParser
        An object with the functionalities of the configparser module

    caldf : pandas DataFrame
        The current scan calibration parameters

    Returns
    -------
    mphnlmprocim : 2D numpy arrray 
        The dose distribution
    """
    
    
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.0.tif')
    
    # Denoise
    udim = nlmf(dosefilename, config)
    
    # Optical density image
    dim = np.log10(2**16/(udim+0.0000001))
    
    # Current multiphase calibration parameters
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values
    
    # Rational approximation
    
    # Define models
    rratfmodel = Model(iratf)
    gratfmodel = Model(iratf)
    bratfmodel = Model(iratf)
    
    # Initialize parameters
    rratparams = rratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    gratparams = gratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    bratparams = bratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    ) 
    
    
    # Generate calibration points
    
    vDrat = np.array([0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4., 5., 7., 9.])
    
    vdrrat =  calf(vDrat, *rcalps)
    vdgrat =  calf(vDrat, *gcalps)
    vdbrat =  calf(vDrat, *bcalps)
    
    # Fit
    rratfit = rratfmodel.fit(data=vDrat, params=rratparams, d=vdrrat)
    gratfit = gratfmodel.fit(data=vDrat, params=gratparams, d=vdgrat)
    bratfit = bratfmodel.fit(data=vDrat, params=bratparams, d=vdbrat)
    
    # Rational calibration paramters
    rratps = np.array([k.value for k in rratfit.params.values()])
    gratps = np.array([k.value for k in gratfit.params.values()])
    bratps = np.array([k.value for k in bratfit.params.values()])
    
    # Dose calculation
    #adDr=Ricalfv(dim[...,0], rcalps, rratps)
    #adDg=Gicalfv(dim[...,1], gcalps, gratps)
    #adDb=Bicalfv(dim[...,2], bcalps, bratps)
    print('Red channel dose calculation:')
    adDr = np.array([[Ricalf(e, rcalps, rratps) for e in r] for r in tqdm(dim[...,0])])
    print('Green channel dose calculation:')
    adDg = np.array([[Gicalf(e, gcalps, gratps) for e in r] for r in tqdm(dim[...,1])])
    print('Blue channel dose calculation:')
    adDb = np.array([[Bicalf(e, bcalps, bratps) for e in r] for r in tqdm(dim[...,2])])
    
    Dmax = float(config['DosePlane']['Dmax'])
    wr, wg, wb = float(config['NonLocalMeans']['wRed']), float(config['NonLocalMeans']['wGreen']), float(config['NonLocalMeans']['wBlue'])
    wT = wr + wg + wb

    mphnlmprocim = (wr*adDr + wg*adDg + wb*adDb)/wT

    mphnlmprocim = np.nan_to_num(mphnlmprocim, posinf=1e10, neginf=-1e10)

    mphnlmprocim[mphnlmprocim < 0] = 0

    mphnlmprocim[mphnlmprocim > Dmax] = Dmax

    # Return the dose image
    return mphnlmprocim
    
def mayermltchprocf(imfile=None, config=None, caldf=None, ccdf=None):
    """
    A function to process the dose distribution image using the Mayer implementation of the Micke multichannel method
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    config : ConfigParser
        An object with the functionalities of the configparser module

    caldf : pandas DataFrame
        The current scan calibration parameters
        
    ccdf : pandas DataFrame
        A data structure containing the relevant geometric parameters for the spatial correction

    Returns
    -------
    mayermltchprocim : 2D numpy arrray 
        The dose distribution
    """
    
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.Film.tif')
    
    # Read the scanned dose image, and split the digital signal of each channel
    im = imread(dosefilename)
    Rim = im[..., 0]
    Gim = im[..., 1]
    Bim = im[..., 2]
    
    # Current multiphase calibration parameters
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values

    # Background signal for every color channel

    SbR, SbG, SbB = 2**16/10**rcalps[0], 2**16/10**gcalps[0], 2**16/10**bcalps[0]
    
    # Rational approximation
    
    # Define models
    rratfmodel = Model(iratf)
    gratfmodel = Model(iratf)
    bratfmodel = Model(iratf)
    
    # Initialize parameters
    rratparams = rratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    gratparams = gratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    bratparams = bratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    ) 
    
    
    # Generate calibration points
    
    vDrat = np.array([0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4., 5., 7., 9.])
    
    vdrrat =  calf(vDrat, *rcalps)
    vdgrat =  calf(vDrat, *gcalps)
    vdbrat =  calf(vDrat, *bcalps)
    
    # Fit
    rratfit = rratfmodel.fit(data=vDrat, params=rratparams, d=vdrrat)
    gratfit = gratfmodel.fit(data=vDrat, params=gratparams, d=vdgrat)
    bratfit = bratfmodel.fit(data=vDrat, params=bratparams, d=vdbrat)
    
    # Rational calibration paramters
    aR, bR, cR = [k.value for k in rratfit.params.values()]
    aG, bG, cG = [k.value for k in gratfit.params.values()]
    aB, bB, cB = [k.value for k in bratfit.params.values()]
    
    # Dose calculation
    print('Dose calculation (Mayer implementation multichannel algorithm):')
    adDr = np.zeros_like(dim[...,0])
    adDg = np.zeros_like(dim[...,1])
    adDb = np.zeros_like(dim[...,2])
    nrs = dim.shape[1]
    for j in tqdm(np.arange(nrs)):
        xc = np.abs(ccdf.o - ccdf.c) * ccdf.s / 25.4
        x = np.abs(ccdf.o - (ccdf.p0 + j)) * ccdf.s / 25.4
        npx = dim.shape[0]
        for i in np.arange(npx):
            
            # Red channel
            f, phir, kr, phib, kb = rcalps
            rcalcps = np.array([f, phir * phiRrf(x)/phiRrf(xc), kr, phib * phiRbf(x)/phiRbf(xc), kb])
            adDr[i, j] = Ricalf(dim[i, j, 0], rcalcps, rratps)
            
            # Green channel
            f, phir, kr, phib, kb = gcalps
            gcalcps = np.array([f, phir * phiGrf(x)/phiGrf(xc), kr, phib * phiGbf(x)/phiGbf(xc), kb])
            adDg[i, j] = Gicalf(dim[i, j, 1], gcalcps, rratps)
            
            # Blue channel
            f, phir, kr, phib, kb = bcalps
            bcalcps = np.array([f, phir * phiBrf(x)/phiBrf(xc), kr, phib * phiBbf(x)/phiBbf(xc), kb])
            adDb[i, j] = Bicalf(dim[i, j, 2], bcalcps, rratps)
    
    Dmax = float(config['DosePlane']['Dmax'])
    wr, wg, wb = float(config['NonLocalMeans']['wRed']), float(config['NonLocalMeans']['wGreen']), float(config['NonLocalMeans']['wBlue'])
    wT = wr + wg + wb

    mphspcnlmprocim = (wr*adDr + wg*adDg + wb*adDb)/wT

    mphspcnlmprocim = np.nan_to_num(mphspcnlmprocim, posinf=1e10, neginf=-1e10)

    mphspcnlmprocim[mphspcnlmprocim < 0] = 0

    mphspcnlmprocim[mphspcnlmprocim > Dmax] = Dmax

    # Return the dose image
    return mphspcnlmprocim
    
def mphspcnlmprocf(imfile=None, config=None, caldf=None, ccdf=None):
    """
    A function to process the dose distribution image using nonlocal means denoising and the multiphase calibration model with spatial correction
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    config : ConfigParser
        An object with the functionalities of the configparser module

    caldf : pandas DataFrame
        The current scan calibration parameters
        
    ccdf : pandas DataFrame
        A data structure containing the relevant geometric parameters for the spatial correction

    Returns
    -------
    mphspcnlmprocim : 2D numpy arrray 
        The dose distribution
    """
    
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.Film.tif')
    
    # Denoise
    udim = nlmf(dosefilename, config)
    
    # Optical density image
    dim = np.log10(2**16/(udim+0.0000001))
    
    # Current multiphase calibration parameters
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values
    
    # Spatial correction functions
    recadc = np.load(config['Models']['oadcFile'], allow_pickle=True)
    phiRrf = recadc[0, 0].item()
    phiGrf = recadc[0, 1].item()
    phiBrf = recadc[0, 2].item()
    phiRbf = recadc[1, 0].item()
    phiGbf = recadc[1, 1].item()
    phiBbf = recadc[1, 2].item()
    
    # Rational approximation
    
    # Define models
    rratfmodel = Model(iratf)
    gratfmodel = Model(iratf)
    bratfmodel = Model(iratf)
    
    # Initialize parameters
    rratparams = rratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    gratparams = gratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    bratparams = bratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    ) 
    
    
    # Generate calibration points
    
    vDrat = np.array([0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4., 5., 7., 9.])
    
    vdrrat =  calf(vDrat, *rcalps)
    vdgrat =  calf(vDrat, *gcalps)
    vdbrat =  calf(vDrat, *bcalps)
    
    # Fit
    rratfit = rratfmodel.fit(data=vDrat, params=rratparams, d=vdrrat)
    gratfit = gratfmodel.fit(data=vDrat, params=gratparams, d=vdgrat)
    bratfit = bratfmodel.fit(data=vDrat, params=bratparams, d=vdbrat)
    
    # Rational calibration paramters
    rratps = np.array([k.value for k in rratfit.params.values()])
    gratps = np.array([k.value for k in gratfit.params.values()])
    bratps = np.array([k.value for k in bratfit.params.values()])
    
    # Dose calculation
    print('Dose calculation:')
    adDr = np.zeros_like(dim[...,0])
    adDg = np.zeros_like(dim[...,1])
    adDb = np.zeros_like(dim[...,2])
    nrs = dim.shape[1]
    for j in tqdm(np.arange(nrs)):
        xc = np.abs(ccdf.o - ccdf.c) * ccdf.s / 25.4
        x = np.abs(ccdf.o - (ccdf.p0 + j)) * ccdf.s / 25.4
        npx = dim.shape[0]
        for i in np.arange(npx):
            
            # Red channel
            f, phir, kr, phib, kb = rcalps
            rcalcps = np.array([f, phir * phiRrf(x)/phiRrf(xc), kr, phib * phiRbf(x)/phiRbf(xc), kb])
            adDr[i, j] = Ricalf(dim[i, j, 0], rcalcps, rratps)
            
            # Green channel
            f, phir, kr, phib, kb = gcalps
            gcalcps = np.array([f, phir * phiGrf(x)/phiGrf(xc), kr, phib * phiGbf(x)/phiGbf(xc), kb])
            adDg[i, j] = Gicalf(dim[i, j, 1], gcalcps, rratps)
            
            # Blue channel
            f, phir, kr, phib, kb = bcalps
            bcalcps = np.array([f, phir * phiBrf(x)/phiBrf(xc), kr, phib * phiBbf(x)/phiBbf(xc), kb])
            adDb[i, j] = Bicalf(dim[i, j, 2], bcalcps, rratps)
    
    Dmax = float(config['DosePlane']['Dmax'])
    wr, wg, wb = float(config['NonLocalMeans']['wRed']), float(config['NonLocalMeans']['wGreen']), float(config['NonLocalMeans']['wBlue'])
    wT = wr + wg + wb

    mphspcnlmprocim = (wr*adDr + wg*adDg + wb*adDb)/wT

    mphspcnlmprocim = np.nan_to_num(mphspcnlmprocim, posinf=1e10, neginf=-1e10)

    mphspcnlmprocim[mphspcnlmprocim < 0] = 0

    mphspcnlmprocim[mphspcnlmprocim > Dmax] = Dmax

    # Return the dose image
    return mphspcnlmprocim
    
def premphspcnlmprocf(imfile=None, config=None, caldf=None, ccdf=None):
    """
    A function to preprocess the dose distribution image using nonlocal means denoising and the multiphase calibration model with spatial correction
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    config : ConfigParser
        An object with the functionalities of the configparser module

    caldf : pandas DataFrame
        The current scan calibration parameters
        
    ccdf : pandas DataFrame
        A data structure containing the relevant geometric parameters for the spatial correction

    Returns
    -------
    mphspcnlmprocim : 2D numpy arrray 
        The dose distribution
    """
    print('Dose preprocessing...')
    
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.Film.tif')
    
    # Denoise
    udim = nlmf(dosefilename, config)
    
    # Optical density image
    dim = np.log10(2**16/(udim+0.0000001))
    
    # Current multiphase calibration parameters
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values
    
    # Spatial correction functions
    recadc = np.load(config['Models']['oadcFile'], allow_pickle=True)
    phiRrf = recadc[0, 0].item()
    phiGrf = recadc[0, 1].item()
    phiBrf = recadc[0, 2].item()
    phiRbf = recadc[1, 0].item()
    phiGbf = recadc[1, 1].item()
    phiBbf = recadc[1, 2].item()
    
    # Rational approximation
    
    # Define models
    rratfmodel = Model(iratf)
    gratfmodel = Model(iratf)
    bratfmodel = Model(iratf)
    
    # Initialize parameters
    rratparams = rratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    gratparams = gratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    bratparams = bratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    ) 
    
    
    # Generate calibration points
    
    vDrat = np.array([0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4., 5., 7., 9.])
    
    vdrrat =  calf(vDrat, *rcalps)
    vdgrat =  calf(vDrat, *gcalps)
    vdbrat =  calf(vDrat, *bcalps)
    
    # Fit
    rratfit = rratfmodel.fit(data=vDrat, params=rratparams, d=vdrrat)
    gratfit = gratfmodel.fit(data=vDrat, params=gratparams, d=vdgrat)
    bratfit = bratfmodel.fit(data=vDrat, params=bratparams, d=vdbrat)
    
    # Rational calibration paramters
    rratps = np.array([k.value for k in rratfit.params.values()])
    gratps = np.array([k.value for k in gratfit.params.values()])
    bratps = np.array([k.value for k in bratfit.params.values()])
    
    print('Dose calculation:')
    DimcolsList = []
    dimcols = [dim[:, y, :] for y in np.arange(dim.shape[1])]
    xc = np.abs(ccdf.o - ccdf.c) * ccdf.s / 25.4
    
    Dim = np.array(
        list(
            tqdm(
                map(wrapped_colDoseCalculationMphspcnlmprocf, 
                    [
                        [col, dimcol, ccdf, xc, rcalps, gcalps, bcalps,
                        phiRrf, phiRbf, phiGrf, phiGbf, phiBrf, phiBbf,
                        rratps, gratps, bratps] for col, dimcol in enumerate(dimcols)
                    ]
                ), total=len(dimcols)
            )
        )     
    )
    return Dim    

def mphspcnlmprocf_multiprocessing(imfile=None, config=None, caldf=None, ccdf=None):
    """
    A function to preprocess the dose distribution image using nonlocal means denoising and the multiphase calibration model with spatial correction
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the image file, the file containing the scanned image of the dose distribution, the calibration strip and the base strip in TIFF format.
        
    config : ConfigParser
        An object with the functionalities of the configparser module

    caldf : pandas DataFrame
        The current scan calibration parameters
        
    ccdf : pandas DataFrame
        A data structure containing the relevant geometric parameters for the spatial correction

    Returns
    -------
    mphspcnlmprocim : 2D numpy arrray 
        The dose distribution
    """
    print('Dose preprocessing...')
    
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.Film.tif')
    
    # Denoise
    udim = nlmf(dosefilename, config)
    
    # Optical density image
    dim = np.log10(2**16/(udim+0.0000001))
    
    # Current multiphase calibration parameters
    rcalps = caldf.iloc[0].values
    gcalps = caldf.iloc[1].values
    bcalps = caldf.iloc[2].values
    
    # Spatial correction functions
    recadc = np.load(config['Models']['oadcFile'], allow_pickle=True)
    phiRrf = recadc[0, 0].item()
    phiGrf = recadc[0, 1].item()
    phiBrf = recadc[0, 2].item()
    phiRbf = recadc[1, 0].item()
    phiGbf = recadc[1, 1].item()
    phiBbf = recadc[1, 2].item()
    
    # Rational approximation
    
    # Define models
    rratfmodel = Model(iratf)
    gratfmodel = Model(iratf)
    bratfmodel = Model(iratf)
    
    # Initialize parameters
    rratparams = rratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    gratparams = gratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    )
    
    bratparams = bratfmodel.make_params(
        a = 0.1,
        b = 0.1,
        c = 0.1
    ) 
    
    
    # Generate calibration points
    
    vDrat = np.array([0.5, 0.75, 1., 1.25, 1.5, 2., 3., 4., 5., 7., 9.])
    
    vdrrat =  calf(vDrat, *rcalps)
    vdgrat =  calf(vDrat, *gcalps)
    vdbrat =  calf(vDrat, *bcalps)
    
    # Fit
    rratfit = rratfmodel.fit(data=vDrat, params=rratparams, d=vdrrat)
    gratfit = gratfmodel.fit(data=vDrat, params=gratparams, d=vdgrat)
    bratfit = bratfmodel.fit(data=vDrat, params=bratparams, d=vdbrat)
    
    # Rational calibration paramters
    rratps = np.array([k.value for k in rratfit.params.values()])
    gratps = np.array([k.value for k in gratfit.params.values()])
    bratps = np.array([k.value for k in bratfit.params.values()])
    
    
    print('Dose calculation:')
    DimcolsList = []
    dimcols = [dim[:, y, :] for y in np.arange(dim.shape[1])]
    xc = np.abs(ccdf.o - ccdf.c) * ccdf.s / 25.4
    xl = [np.abs(ccdf.o - (ccdf.p0 + col)) * ccdf.s / 25.4 for col, dimcol in enumerate(dimcols)]
    colsrcalps = np.array([rcalps * np.array([1, phiRrf(x)/phiRrf(xc), 1, phiRbf(x)/phiRbf(xc), 1]) for x in xl], dtype=object)
    colsgcalps = np.array([gcalps * np.array([1, phiGrf(x)/phiGrf(xc), 1, phiGbf(x)/phiGbf(xc), 1]) for x in xl], dtype=object)
    colsbcalps = np.array([bcalps * np.array([1, phiBrf(x)/phiBrf(xc), 1, phiBbf(x)/phiBbf(xc), 1]) for x in xl], dtype=object)
    
    with Pool(None) as p:
        Dim = np.array(
            list(
                tqdm(
                    p.imap(wrapped_colDoseCalculationMphspcnlmprocf, 
                              [[dimcol,
                                colsrcalps[col], colsgcalps[col], colsbcalps[col], 
                                rratps, gratps, bratps] for col, dimcol in enumerate(dimcols)]
                    ), total=len(dimcols)
                )
            )
        )
    return Dim    


def colDoseCalculationMphspcnlmprocf(parl):
    """
    A function to calculate the dose for every color channel in every pixel of a given colummn from the optical density image
    It is an accessory function for multiprocessing. It should not be call outside the premphspcnlmprocf function.
    
    ...
    
    Attributes
    ----------
    
    Returns
    -------
    Dimcol : 2D numpy arrray 
        The column dose distribution for the three color channels
    """
       
    dimcol = parl[0]
    colrcalps = parl[1]
    colgcalps = parl[2]
    colbcalps = parl[3]
    rratps = parl[4]
    gratps = parl[5]
    bratps = parl[6]
    
    Dimcol  = np.empty_like(dimcol)
    
    for i in np.arange(len(dimcol)):
        # Red channel
        Dimcol[i, 0] = Ricalf(dimcol[i, 0], colrcalps, rratps)

        # Green channel
        Dimcol[i, 1] = Gicalf(dimcol[i, 1], colgcalps, gratps)

        # Blue channel
        Dimcol[i, 2] = Bicalf(dimcol[i, 2], colbcalps, bratps)
    
    return Dimcol

def wrapped_colDoseCalculationMphspcnlmprocf(parl):
    return colDoseCalculationMphspcnlmprocf(parl)
    
    
def postmphspcnlmprocf(Dim=None, config=None, planfile=''):
    """
    Postprocessing the dose distribution image
    
    ...
    
    Attributes
    ----------
    Dim : 3D numpy array
        A numpy array with the image dose from every color channel
        
    config : ConfigParser
        An object with the functionalities of the configparser module
        
    planfile: string
        The name of the DICOM file with the calculated dose distribution

    Returns
    -------
    mphspcnlmprocim : 2D numpy arrray 
        The dose distribution
    """
    Dmax = float(config['DosePlane']['Dmax'])
    if planfile != '':
        pDim = DICOMDose(planfile)
        Dmax = 1.1 * pDim.max()
        
    wr, wg, wb = float(config['NonLocalMeans']['wRed']), float(config['NonLocalMeans']['wGreen']), float(config['NonLocalMeans']['wBlue'])
    wT = wr + wg + wb
    
    mphspcnlmprocim = (wr*Dim[..., 0] + wg*Dim[..., 1] + wb*Dim[..., 2])/wT

    mphspcnlmprocim = np.nan_to_num(mphspcnlmprocim, posinf=1e10, neginf=-1e10)

    mphspcnlmprocim[mphspcnlmprocim < 0] = 0

    mphspcnlmprocim[mphspcnlmprocim > Dmax] = Dmax

    # Return the dose image
    return mphspcnlmprocim
    
def HeaderCreator(DataOrigin=None, AcqType='Acquired Portal', PatientId1='', PatientId2='', LastName='', FirstName='', pxsp=[], imsz=[]):
    """
    A function to create the heaer of the dxf file
    
    ...
    
    Attributes
    ----------
    DataOrigin : str
        The name of the file of which the data have been extracted
        
    AcqType : str
        The type of the orgin data: acquired or predicted

    PatientId1 : str
        The first patient identification 

    PatientId2 : str
        The second patient identification 

    LastName : str
        The patient family name

    First Name : str
        The patient given name
        
    pxsp : list
        A list with the pixel spacing in mm
    
    imsz : list
        A list with the image size in pixels

    Returns
    -------
    header : str
        The header of the dxf file
    """
    
    headerGeneral = '\n'.join(
        [ '[General]',
          'FileFormat=Generic Dosimetry Exchange Format',
          'Version=1.0',
          'Creator=Film Dosimetry',
          'CreatorVersion=0.1',
          '[Geometry]',
          'Dimensions=2',
          'Axis1=X',
          'Size1=' + str(int(imsz[0])),
          'Res1=' + str(pxsp[0]),
          'Offset1=0.0',
          'Unit1=mm',
          'Separator1=\\t',
          'Axis2=Y',
          'Size2=' + str(int(imsz[1])),
          'Res2=' + str(pxsp[1]),
          'Offset2=-0.0',
          'Unit2=mm',
          'Separator2=\\n',
        ]
    )
    
    headerInterpretation = '\n'.join(
        [ '[Interpretation]',
          'Type=' + AcqType,
          'DataType=%f',
          'Unit=CU',
          'Location=Imager',
          'Medium=Undefined',
        ]
    )
    
    headerPatient = '\n'.join(
        ['[Patient]',
        'PatientId1=' + PatientId1,
        'PatientId2=' + PatientId2,
        'LastName=' + LastName,
        'FirstName=' + FirstName,
        ] 
    )
    
    headerField = '\n'.join(
        ['[Field]',
          'PlanId=QAP',
          'FieldId=Field 1',
          'ExternalBeamId=TrueBeam1',
          'BeamType=Photon',
          'Energy=6000',
          'SAD=100',
          'Scale=IEC1217',
          'GantryAngle=0',
          'CollRtn=0',
          'CollX1=10.0',
          'CollX2=10.0',
          'CollY1=10.0',
          'CollY2=10.0',
        ] 
    )
    
    if DataOrigin:
        datetimestr = datetime.fromtimestamp(os.path.getctime(DataOrigin)).strftime("%m/%d/%Y, %H:%M:%S")
    else:
        datetimestr = ''
    
    headerPortal = '\n'.join(
        ['[PortalDose]',
          'SID=100.0',
          'Date=' + datetimestr,
        ] 
    )
    
    headerData = '\n'.join(
        ['[Data]'
        ]
    )
    
    header = headerGeneral + '\n' + headerInterpretation + '\n' + headerPatient + '\n' + headerField + '\n' + headerPortal + '\n' + headerData + '\n'
    
    return header
    
def dxfWriter(Data=None, dxfFileName='Film.dxf', DataOrigin=None, AcqType='Acquired Portal', PatientId1='', PatientId2='', LastName='', FirstName='', pxsp=[], imsz=[]):
    """
    A function to write the dxf file
    
    ...
    
    Attributes
    ----------
    Data : 2D numpy array
        The data to be exported
        
    dxfFileName : str or Path
        The path file to be written in

    DataOrigin : str
        The name of the file of which the data have been extracted

    AcqType : str
        The type of the orgin data: acquired or predicted

    PatientId1 : str
        The first patient identification 

    PatientId2 : str
        The second patient identification 

    LastName : str
        The patient family name

    First Name : str
        The patient given name

    pxsp : list
        A list with the pixel spacing in mm
    
    imsz : list
        A list with the image size in pixels

     Returns
    -------
        No data returned on exit
    """
    
    with open(dxfFileName, 'w') as dxf:
        # Write the header
        header = HeaderCreator(DataOrigin=DataOrigin, AcqType=AcqType, PatientId1=PatientId1, PatientId2=PatientId2, LastName=LastName, FirstName=FirstName, pxsp=pxsp, imsz=imsz)
        for line in header:
            dxf.write(line)
        
        df = pd.DataFrame(Data)
        # Write the data
        df.to_csv(dxf, sep='\t', header=False, index=False, float_format='%.2f')
    
def DICOMDemographics(dcmfile=None):
    """
    A function to get the patient demographics from the DICOM file
    
    ...
    
    Attributes
    ----------
    dcmfile : str
        The name of the DICOM file with the calculated dose distribution
        

     Returns
    -------
        democdict :  Dictionary
        A dictinary containing the patient demographics
    """
    
    # Read the DICOM file
    dcmf = dicom.read_file(dcmfile)

    # Demographics
    PatientId1 = dcmf.PatientID
    PatientId2 = ''
    PatientName = str(dcmf.PatientName)
    LastName = PatientName.split('^')[0]
    FirstName = PatientName.split('^')[1]
  
    demodict = {'PatientId1' : PatientId1, 'PatientId2' : PatientId2, 'LastName' : LastName, 'FirstName' : FirstName}
    
    return demodict
    
def DICOMPixelSpacing(dcmfile=None):
    """
    A function to get the spatial resolution of the calculated dose distribution from the DICOM file
    
    ...
    
    Attributes
    ----------
    dcmfile : str
        The name of the DICOM file with the calculated dose distribution
        

     Returns
    -------
        pxsp :  List
        A list with the X and Y pixel spacing in mm
    """
    # Read the DICOM file
    dcmf = dicom.read_file(dcmfile)
    
    return dcmf.PixelSpacing
    
def DICOMImageSize(dcmfile=None):
    """
    A function to get the image size of the calculated dose distribution from the DICOM file
    
    ...
    
    Attributes
    ----------
    dcmfile : str
        The name of the DICOM file with the calculated dose distribution
        

     Returns
    -------
        imsz :  List
        A list with the X and Y image size in pixels
    """
    # Read the DICOM file
    dcmf = dicom.read_file(dcmfile)
    
    return [dcmf.Rows, dcmf.Columns]
    
def DICOMDose(dcmfile=None):
    """
    A function to get the calculated dose distribution from the DICOM file
    
    ...
    
    Attributes
    ----------
    dcmfile : str
        The name of the DICOM file with the calculated dose distribution
        

     Returns
    -------
        dose :  2D numpy array
        An array containing the dose distribution
    """
    # Read the DICOM file
    dcmf = dicom.read_file(dcmfile)

    # Read the dose distribution
    Dim = dcmf.pixel_array*dcmf.DoseGridScaling
   
    return Dim

def DoseImageSize(im=None):
    """
    A function to get the image size of the film measured dose distribution 
    
    ...
    
    Attributes
    ----------
    im : 2D numpy array
        The array containing the measured dose distribution

     Returns
    -------
        imsz :  List
        A list with the X and Y image size in pixels
    """
    imsz = [im.shape[1], im.shape[0]]
    return imsz

def TIFFDoseImageSize(imfile=None):
    """
    A function to get the image size of the film measured dose distribution from the TIFF file
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the scan image with the measured dose distribution, the calibration strip and the background patch

     Returns
    -------
        imsz :  List
        A list with the X and Y image size in pixels
    """
    dosefilename = Path(imfile)
    dosefilename = dosefilename.with_suffix('.0.tif')
    with TiffFile(dosefilename) as tif:
        page_tags = tif.pages[0].tags
        
    return [page_tags['ImageLength'].value, page_tags['ImageWidth'].value] 

def TIFFPixelSpacing(imfile=None):
    """
    A function to get the pixel spacing from the TIFF file
    
    ...
    
    Attributes
    ----------
    imfile : str
        The name of the scan image with the measured dose distribution, the calibration strip and the background patch

     Returns
    -------
        pxsp :  List
        A list with the X and Y pixel spacing in mm
    """
    
    with TiffFile(imfile) as tif:
        page_tags = tif.pages[0].tags
    xres = page_tags['XResolution'].value
    yres = page_tags['YResolution'].value
    xdpi = xres[0]/xres[1]
    ydpi = yres[0]/yres[1]
    inch = 25.4 # mm
    return [inch/xdpi, inch/ydpi]
