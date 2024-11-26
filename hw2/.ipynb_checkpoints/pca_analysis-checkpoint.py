from PIL import Image
import numpy as np
import glob

IMAGES_FOLDER = "resized_images/"

def readImagesFlat(imageFolder = IMAGES_FOLDER):
    images = []
    for imageFile in glob.glob(imageFolder + "image_*.png"):
        with Image.open(imageFile) as image:
                images.append(np.array(image))
    images = np.array(images)
    return images.reshape((images.shape[0], -1, images.shape[-1])).astype(np.float32)

def normalizeMean(X):
    return X - X.mean(axis=0)

def covarianceMatrix(X):
    return np.cov(X, rowvar=False)

def findEigens(X):
    eigValues, eigVectors = np.linalg.eigh(X)
    sorted_indices = np.argsort(eigValues)[::-1]
    eigValues = eigValues[sorted_indices]
    eigVectors = eigVectors[:, sorted_indices]
    return eigValues, eigVectors
    
def binarySearch(eigValues, aim):
    prefixSum = eigValues.cumsum()
    left, right, ans = 0, len(eigValues), -1
    while left >= right:
        mid = (left + right) / 2
        if prefixSum[mid] >= aim:
            right = mid - 1
            ans = mid
        else:
            left = mid + 1
    return ans

def calculatePrincipalComponenetsOfRGB(images):
    images = images.reshape((images.shape[-1], *images.shape[:-1]))
    eigValues = []
    eigVectors = []
    for colorChannel in range(3):
        u,v = findEigens(covarianceMatrix(normalizeMean(images[colorChannel])))
        eigValues.append(u)
        eigVectors.append(v)
    eigValues = np.array(eigValues)
    eigVectors = np.array(eigVectors)
    total = eigValues.sum(axis=1)
    print(eigValues[:,:10] / total[np.newaxis,:], eigVectors[:,:10])

    

images = readImagesFlat()
calculatePrincipalComponenetsOfRGB(images)

