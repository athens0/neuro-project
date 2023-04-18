import numpy as np 

from skimage import exposure
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

import matplotlib.pyplot as plt


ped = imread('input/ped2.jpg')
ped = rgb2gray(ped)
imshow(ped)
print(ped.shape)

resized_ped = resize(ped, (240,120)) 
imshow(resized_ped)
print(resized_ped.shape)

hog_features = []
fd, hog_ped = hog(resized_ped, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualize=True)

hog_features.append(fd)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True) 

ax1.imshow(resized_ped, cmap=plt.cm.gray) 
ax1.set_title('Изначальное изображение') 

# Rescale histogram for better display 
hog_ped_rescaled = exposure.rescale_intensity(hog_ped, in_range=(0, 10)) 

ax2.imshow(hog_ped_rescaled, cmap=plt.cm.gray) 
ax2.set_title('Гистограмма направленных градиентов (HOG)')

# store to file
plt.savefig("ped2_hog.png", dpi=125)

plt.show()


"""clf = svm.SVC()
hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features,labels))
np.random.shuffle(data_frame)

percentage = 80
partition = int(len(hog_features)*percentage/100)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))"""