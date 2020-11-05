import cv2
import numpy as np
import pandas as pd

def predict(model, image_id):
    # load the image
    submission = []
    image = cv2.imread(f'{image_id}') # reads the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grey scale
    image = cv2.resize(image, (256,256))
    norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = np.array([[norm_image]])
    image = torch.from_numpy(image).cuda()

    out = model(image)

    blobs = out[0][0].detach().cpu().numpy()
    thresh  = np.uint8(np.where(blobs > 0.5, 255, 0))
    kernel_ones = np.ones((3,3),np.uint8)
    kernel_gauss = 1/273 * np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7], [4,16,26,16,4], [1,4,7,4,1]],np.uint8 )
    kernel_laplace = np.array([[-1,-1,-1],[-1,8,-1], [-1,-1,-1]],np.uint8 )

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_gauss, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_laplace, iterations=2)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel_ones, iterations=3)
    
    countours ,_ = cv2.findContours(opening.astype(int),cv2.RETR_FLOODFILL,cv2.CHAIN_APPROX_SIMPLE)
    s = ""
    for c in countours:
        x,y,w,h = cv2.boundingRect(c)
        if x <= 0 :
            x = 0
        if y <= 0 :
            y = 0
        if w * h > 70 :
            s += "1.0 " + str(x * 4) + " " + str(y * 4) + " " + str(h* 4 ) + " " + str(w* 4 )+ " "
    return s
