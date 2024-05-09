import cv2
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from picamera2 import Picamera2, Preview
import time
import paho.mqtt.client as mqtt  
import io
import base64


c= np.array([[ 1.00715201e+00, -2.32810839e-02, -3.88794829e+01], [7.54220558e-03,  9.75861385e-01,  2.86827547e+01], [3.51586112e-05, -7.40422299e-05,  1.00000000e+00]])
global im_color


cam0=Picamera2(0)
cam1=Picamera2(1)

#set camaras
cam0.preview_configuration.main.size = (700,500)
cam1.preview_configuration.main.size = (700,500)
cam0.preview_configuration.main.format = "XBGR8888"
cam1.preview_configuration.main.format = "XBGR8888"
cam0.preview_configuration.align()
cam1.preview_configuration.align()
cam0.configure("preview")
cam1.configure("preview")
cam0.start()
cam1.start()

# Create Sift object :3
sift = cv2.SIFT_create()

# radio, es el tamaño de la lente en centimetros
class Correccion: 
            
    def img_alignment_sequoia(self, img_base_NIR, img_RED, width, height):    
            global c 
            # Resize the images to the same size specified in image_SIZE
            base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
            b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)

            # Stabilize the RED image with respect to the NIR base image
            stb_RED =cv2.warpPerspective(b_RED,c,(width, height))
            
            return base_NIR, stb_RED


#--------------------------------------------------------------------------------------------------------------------
#[Configuracion de lectura]
def on_message(client, userdata, message):
    mes = str(message.payload.decode("utf-8"))
    comands(mes)

#-----------------------------------------------------------------------------------------------------------------------------------
#[Configuracion de la comunicacion]
#Setup de mqtt
comand = "no data"
broker_address="localhost"            #Broker address
client = mqtt.Client("OjosDeQuetzal")   #Nombre del dispositivo
client.connect(broker_address)          #connect to clint
client.loop_start()                     #start loop
client.on_message=on_message            #attach function to callback lectura
client.subscribe("data")                #Tópico que se suscribe
#-----------------------------------------------------------------------------------------------------------------------------------
#[Conversor de Imagen]  
def send_IMG(img):
    buf = io.BytesIO()
    plt.imsave(buf, img, format='png')  
    image_data = buf.getvalue()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    client.publish("Imagen", encoded_image)
#--------------------------------------------------------------------------------------------------------------------
#[Comandos de imagen]
def comands(number):
    global im_color 
    NIR = cv2.imread(r"/home/kouriakova/Ojos de Quetzal/ojos_quetzal/IMG_700101_001240_0000_NIR.TIF", 0)
    RED = cv2.imread(r"/home/kouriakova/Ojos de Quetzal/ojos_quetzal/IMG_700101_001240_0000_RED.TIF", 0)
    NDVI = im_color
    """
    RGB = cv2.imread(url_img_RED, 0)
    RE = cv2.imread(url_img_RED, 0)
    GREEN = cv2.imread(url_img_RED, 0)
    """
    for i in range (6):
        if(i == int(number)):
            if(i == 1):
                send_IMG(NIR)
            elif(i == 2):
                send_IMG(RED)
            elif(i == 3):
                send_IMG(NDVI)
"""
            elif(i == 4):
                send_IMG(NIR)
            elif(i == 5):
                send_IMG(NIR)
            elif(i == 6):
                send_IMG(NIR)"""
#-----------------------------------------------------------------------------------------------------------------------------------
#[Clase para calcular el índice NDVI]
class NDVICalculator:
    def ndvi_calculation(self, url_img_RED, url_img_NIR, width=700, height=500):
        img_RED = cv2.imread(url_img_RED, 0)
        img_NIR = cv2.imread(url_img_NIR, 0)

        img_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        img_NIR = cv2.resize(img_NIR, (width, height), interpolation=cv2.INTER_LINEAR)

        red = np.array(img_RED, dtype=float)
        nir = np.array(img_NIR, dtype=float)

        check = np.logical_and(red > 1, nir > 1)

        ndvi = np.where(check, (nir - red) / (nir + red), 0)
        matrix = ndvi * (-1)
        Valor = np.nanmean(matrix, where=matrix > 0)
        #Valor = Valor * (-1)

        if ndvi.min() < 0:
            ndvi = ndvi + (ndvi.min() * -1)

        ndvi = (ndvi * 255) / ndvi.max()
        ndvi = ndvi.round()

        ndvi_image = np.array(ndvi, dtype=np.uint8)

        return ndvi_image, Valor

#-----------------------------------------------------------------------------------------------------------------------------------


while True:
    try:
        global im_color

        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()

        pframe0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        pframe1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

        correccion_img = Correccion()
        calculator = NDVICalculator()

        #define image size
        width = 700
        height = 500

        # Reading images
        Img_RED = pframe0
        Img_NIR = pframe1

        # Create a BGR image with the red and nir
        merged_fix_bad = cv2.merge((Img_RED,Img_RED,Img_NIR)) # First image, misaligned
        merged_fix_bad = cv2.resize(merged_fix_bad, (700, 500), interpolation=cv2.INTER_LINEAR)
        # Assuming merged_fix_bad is supposed to be an RGB image


        # img_base_NIR, img_RED, width, height
        try:   
            stb_NIR, stb_RED =  correccion_img.img_alignment_sequoia(Img_NIR, Img_RED, width, height)
            merged_fix_stb = cv2.merge((stb_RED,stb_RED, stb_NIR))
            merged_fix_stb = merged_fix_stb[50:500, 0:650]
            
            ndvi_image, Valor = calculator.ndvi_calculation(stb_RED,stb_NIR)
            client.publish("NDVI", Valor)
            print(Valor)
            #rotate ndvi_image
            ndvi_image = np.flipud(ndvi_image)
            "Se pinta la imagen con colormap de OpenCV. En mi caso, RAINBOW fue la mejor opción"
            im_color = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)
            im_color = cv2.flip(im_color, 1)
            send_IMG(im_color)

            cv2.imshow('merge', merged_fix_stb)
            key = cv2.waitKey(1) & 0xFF
            if key==ord('q'):
                cv2.destroyAllWindows()
                break
        
        except:
            merged_fix_stb=merged_fix_bad
            merged_fix_stb = merged_fix_stb[50:500, 0:650]
            client.publish("cuidado", -50)
            print(-50)
            send_IMG(merged_fix_bad)

            if key==ord('q'):
                cv2.destroyAllWindows()
                break
        

        



    except:
        print("STOP")

cv2.destroyAllWindows()

