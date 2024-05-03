import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

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
        Valor = ndvi

        if ndvi.min() < 0:
            ndvi = ndvi + (ndvi.min() * -1)

        ndvi = (ndvi * 255) / ndvi.max()
        ndvi = ndvi.round()

        ndvi_image = np.array(ndvi, dtype=np.uint8)

        return ndvi_image, Valor

calculator = NDVICalculator()
ndvi_image, Valor = calculator.ndvi_calculation("IMG_700101_001324_0000_RED.tif", "IMG_700101_001324_0000_NIR.tif")
print(Valor)

#rotate ndvi_image
ndvi_image = np.flipud(ndvi_image)
"Se pinta la imagen con colormap de OpenCV. En mi caso, RAINBOW fue la mejor opción"
im_color = cv2.applyColorMap(ndvi_image, cv2.COLORMAP_JET)
im_color = cv2.flip(im_color, 1)
plt.imshow(im_color)
plt.title('NDVI')
plt.show()


#Respecto a github
"""
todo se puede hacer sin la terminal

"""
