import cv2
import numpy as np
import os.path
import sys
import matplotlib.pyplot as plt
from picamera2 import Picamera2, Preview
import time


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
    def estabilizador_imagen(self, imagen_base, imagen_a_estabilizar, radio=0.75 , error_reproyeccion=4.0, coincidencias=False):     
                
                """Esta clase devuelve una secuencia de imágenes tomadas de la cámara estabilizada con respecto a la primera imagen"""
                
                # Se obtienen los puntos deinterés

                (kpsBase, featuresBase) = self.obtener_puntos_interes(imagen_base)
                (kpsAdicional, featuresAdicional) = self.obtener_puntos_interes(imagen_a_estabilizar)
                # Se buscan las coincidencias        

                M = self.encontrar_coincidencias(imagen_base, imagen_a_estabilizar, kpsBase, kpsAdicional, featuresBase,
                                                featuresAdicional, radio)

                if M is None:
                    print("pocas coincidencias")
                    return None

                if len(M) > 4:
                    # construct the two sets of points

                    #            M2 = cv2.getPerspectiveTransform(ptsA,ptsB)
                    (H, status) = self.encontrar_H_RANSAC_Estable(M, kpsBase, kpsAdicional, error_reproyeccion)
                    estabilizada = cv2.warpPerspective(imagen_base, H, (imagen_base.shape[1], imagen_base.shape[0]))
                    return estabilizada
                print("sin coincidencias")
                return None

            
    def img_alignment_sequoia(self, img_base_NIR, img_RED, width, height):    
            """This class takes two images given by Sequoia Camera and makes a photogrammetric
            alignment. Returns two images (RED, NIR) aligned with each other"""

            # Resize the images to the same size specified in image_SIZE
            base_NIR = cv2.resize(img_base_NIR, (width, height), interpolation=cv2.INTER_LINEAR)
            b_RED = cv2.resize(img_RED, (width, height), interpolation=cv2.INTER_LINEAR)

            # Stabilize the RED image with respect to the NIR base image
            stb_RED = self.estabilizador_imagen(b_RED, base_NIR)

            return base_NIR, stb_RED



    # --------------------------------------------------------------------------
    def obtener_puntos_interes(self, imagen):
        """Se obtienen los puntos de interes cn SIFT"""

        #descriptor = cv2.xfeatures2d.SIFT_create() 
        (kps, features) = sift.detectAndCompute(imagen, None)

        return kps, features

    # --------------------------------------------------------------------------
    def encontrar_coincidencias(self, img1, img2, kpsA, kpsB, featuresA, featuresB, ratio):
        """Metodo para estimar la homografia"""
        #alv003 oye aqui ponle un try y acaba con un except
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        #
        #        # loop over the raw matches
        for m in rawMatches:
            #            # ensure the distance is within a certain ratio of each
            #            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        #        print (matches)
        return matches

    # --------------------------------------------------------------------------
    def encontrar_H_RANSAC(self, matches, kpsA, kpsB, reprojThresh):
        """Metodo para aplicar una H a una imagen y obtener la proyectividad"""

        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (H, status)

        # otherwise, no homograpy could be computed
        return None

    # --------------------------------------------------------------------------
    def encontrar_H_RANSAC_Estable(self, matches, kpsA, kpsB, reprojThresh):
        """Metodo para aplicar una H a una imagen y obtener la proyectividad"""

        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return (H, status)

        return None




while True:
    try:
        frame0 = cam0.capture_array()
        frame1 = cam1.capture_array()

        pframe0 = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
        pframe1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

        correccion_img = Correccion()
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
        
        except:
            merged_fix_stb=merged_fix_bad

        #time.sleep(.1)
        # Use matplotlib to show the images
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(merged_fix_bad)
        #plt.subplot(1, 2, 2)
        #plt.imshow(merged_fix_stb)


        #plt.title('Aligned image')
        #plt.show()

        #'cv2.imshow('cam0',pframe0)
        #cv2.imshow('cam1',pframe1)
        cv2.imshow('merge', merged_fix_stb)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            cv2.destroyAllWindows()
            break
    except KeyboardInterrupt:
        print("STOP")

cv2.destroyAllWindows()