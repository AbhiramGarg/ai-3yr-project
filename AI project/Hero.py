import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ADMIN\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

image = cv2.imread('images\car7.png')
image = imutils.resize(image,width = 500)

cv2.imshow('original',image)

cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('greyscale',gray)
cv2.waitKey(0)

gray = cv2.bilateralFilter(gray,11,17,17)
cv2.imshow('Smooth image',gray)
cv2.waitKey(0)

edge = cv2.Canny(gray,170,200)
cv2.imshow('canny image',edge)
cv2.waitKey(0)

cns , new = cv2.findContours(edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
imagecnt = image.copy()
cv2.drawContours(imagecnt,cns,-1,(0,255,0),3)
cv2.imshow('Contour image',imagecnt)
cv2.waitKey(0)

cns = sorted(cns,key = cv2.contourArea , reverse=True)[:30]
Numberpltcnt = None

image2 = image.copy()
cv2.drawContours(image2,cns,-1,(0,255,0),3)
cv2.imshow('coutour top',image2)
cv2.waitKey(0)
name = 1

for i in cns:
    perimeter = cv2.arcLength(i,True)
    approx = cv2.approxPolyDP(i,0.02*perimeter,True)
    if(len(approx)==4):
        Numberpltcnt  = approx
        x , y, w, h = cv2.boundingRect(i)
        crp_img = image[y:y+h , x:x+w]
        cv2.imwrite(str(name)+'.png',crp_img)
        name+=1
        break

cv2.drawContours(image,[Numberpltcnt],-1,(0,255,0),3)
cv2.imshow('Final image',image)
cv2.waitKey(0)

crp_img_final = '1.png'
cv2.imshow('final crop img',cv2.imread(crp_img_final))
text = pytesseract.image_to_string(crp_img_final,lang = 'eng')
print('Number is :',text)
cv2.waitKey(0)




