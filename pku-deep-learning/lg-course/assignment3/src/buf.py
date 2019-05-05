import cv2
pic = cv2.imread('/Users/k/Desktop/results_imgs/1.png')
pic = pic[0:100,20:100]
cv2.imshow('', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()
