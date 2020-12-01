import cv2
from scipy.fftpack import fftn
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('131317.jpg',0)



cap = cv2.VideoCapture('vtest.mp4')

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

self_out = cv2.VideoWriter('self_output.mp4',fourcc, 20,(frame_width,frame_height),True)
per_frame_color_fft_out = cv2.VideoWriter('per_frame_color_fft_output.mp4', fourcc, 20,(frame_width,frame_height),True)


per_frame_grey_fft_out = cv2.VideoWriter('per_frame_grey_fft_output.mp4', fourcc, 20,(frame_width,frame_height),True)



t = 0
while(cap.isOpened()) and t < 50:
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        gray_dft = cv2.dft(np.float32(gray),flags = cv2.DFT_COMPLEX_OUTPUT)
        gray_dft_shift = np.fft.fftshift(gray_dft)
        gray_magnitude_spectrum = 20*np.log(cv2.magnitude(gray_dft_shift[:,:,0],gray_dft_shift[:,:,1]))

        print(t)
        t+=1
        self_out.write(frame)
        per_frame_color_fft_out.write(magnitude_spectrum)
        per_frame_grey_fft_out.write(gray_magnitude_spectrum)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()






# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()





#f = np.fft.fft2(frame)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))
#magnitude_spectrum *= 255.0/magnitude_spectrum.max()
#print(np.unique(magnitude_spectrum))
#cv2.imshow('magnitude_spectrum',magnitude_spectrum)
#t+=1



#f = np.fft.fft2(img)
#fshift = np.fft.fftshift(f)
#magnitude_spectrum = 20*np.log(np.abs(fshift))

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()


