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


#Matriz del alineacion de la imagen
#c= np.array([[ 1.00715201e+00, -2.32810839e-02, -3.88794829e+01], [7.54220558e-03,  9.75861385e-01,  2.86827547e+01], [3.51586112e-05, -7.40422299e-05,  1.00000000e+00]])
c=np.array([[ 9.94024085e-01, -2.97656751e-02, -3.60910470e+01], [-9.76327155e-03,  9.67355238e-01,  3.71420592e+01], [ 1.60864403e-05, -9.07830029e-05,  1.00000000e+00]] ) 

#variables globales
IO=False
opcion=0

#Iniciar camaras
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

# Create Sift object 
sift = cv2.SIFT_create()

#-----------------------------------------------------------------------------------------------------------------------------------
#Alineacion de imagen
class Correccion: 
            
    def img_alignment_sequoia(self, img_base_NIR, img_RED, width, height):    
            global c 
            # Resize the images to the same size specified in image_SIZE
            base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
            b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)

            # Stabilize the RED image with respect to the NIR base image
            stb_RED =cv2.warpPerspective(b_RED,c,(width, height))
            
            return base_NIR, stb_RED

#-----------------------------------------------------------------------------------------------------------------------------------
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
    #client.loop_stop()    

#--------------------------------------------------------------------------------------------------------------------
#[Comandos de imagen]
def comands(number):
    global stb_NIR
    global stb_RED
    global im_color
    global IO 
    global opcion

    NIR = stb_NIR
    RED = stb_RED
    NDVI = im_color


    for i in range (10):  #rango de 0-9
        if(i == int(number)):
            if(i == 1):
                send_IMG(NIR)
                opcion=1
            elif(i == 2):
                send_IMG(RED)
                opcion=2
            elif(i == 3):
                send_IMG(NDVI)
                opcion=3
#Comandos de posiscion



#Comandos de encendido y apagado
            elif(i == 8):
                IO=False
                opcion=8
                
            elif(i == 9):
                IO=True
                opcion=9
            


#-----------------------------------------------------------------------------------------------------------------------------------
#[Clase para calcular el índice NDVI]
class NDVICalculator:
    def ndvi_calculation(self, img_RED, img_NIR, width=700, height=500):

        img_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)
        img_NIR = cv2.resize(img_NIR, (width, height), interpolation=cv2.INTER_LINEAR)

        red = np.array(img_RED, dtype=float)
        nir = np.array(img_NIR, dtype=float)

        check = np.logical_and(red > 1, nir > 1)
        ndvi = np.where(check, (nir - red) / (nir + red), 0) 
        
        matrix = ndvi
        ro=matrix.max()-matrix.min()
        rd=2
        
        Valor = round(np.mean(matrix),3)
        
        vd=((Valor-matrix.min())*rd/ro)-1
        vd=round(vd,2)

        #print(vd) #validar    

        if ndvi.min() < 0:
            ndvi = ndvi + (ndvi.min() * -1)

        ndvi = (ndvi * 255) / ndvi.max()
        ndvi = ndvi.round()
        ndvi_image = np.array(ndvi, dtype=np.uint8)

        return ndvi_image, vd

#-----------------------------------------------------------------------------------------------------------------------------------
   
while True:

    frame0 = cam0.capture_array()
    frame1 = cam1.capture_array()

    pframe0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    pframe1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    #try:
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

    stb_NIR, stb_RED =  correccion_img.img_alignment_sequoia(Img_NIR, Img_RED, width, height)
    merged_fix_stb = cv2.merge((stb_RED,stb_RED, stb_NIR))
    ndvi_image, Valor = calculator.ndvi_calculation(stb_RED,stb_NIR)
    client.publish("NDVI", Valor)
    #rotate ndvi_image
    #ndvi_image = np.flipud(ndvi_image)
    "Se pinta la imagen con colormap de OpenCV. En mi caso, RAINBOW fue la mejor opción"
    im_color = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_RAINBOW)
    #im_color = cv2.flip(im_color, 1)
    im_color = im_color[50:500, 0:650] 

    if (IO==True)&(opcion==3): #NDVI
        send_IMG(im_color)

    elif(IO==True)&(opcion==2): #RED
        send_IMG(stb_RED)
    
    elif(IO==True)&(opcion==1): #NIR
        send_IMG(stb_NIR)
    
    


#-----------------------------------------------------------------------------------------------------------------------------------
#[Main]

#while

