import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import paho.mqtt.client as mqtt  
import io
import base64

global im_color
#--------------------------------------------------------------------------------------------------------------------
#[Configuracion de lectura]
def on_message(client, userdata, message):
    mes = str(message.payload.decode("utf-8"))
    comands(mes)

#-----------------------------------------------------------------------------------------------------------------------------------
#[Configuracion de la comunicacion]
#Setup de mqtt
comand = "no data"
broker_address="10.25.91.11"            #Broker address
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
    NIR = cv2.imread("IMG_700101_001240_0000_NIR.tif", 0)
    RED = cv2.imread("IMG_700101_001240_0000_RED.tif", 0)
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
#[Main]
def main():
    global im_color
    calculator = NDVICalculator()
    ndvi_image, Valor = calculator.ndvi_calculation("IMG_700101_001240_0000_RED.tif", "IMG_700101_001240_0000_NIR.tif")
    client.publish("NDVI", Valor)
    print(Valor)

    #rotate ndvi_image
    ndvi_image = np.flipud(ndvi_image)
    "Se pinta la imagen con colormap de OpenCV. En mi caso, RAINBOW fue la mejor opción"
    im_color = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)
    im_color = cv2.flip(im_color, 1)
    plt.imshow(im_color)
    send_IMG(im_color)
    plt.title('NDVI')
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------------------
#[Loop]
if __name__ == "__main__":
    main()