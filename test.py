import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/driving_log.csv') as csvfile:
#with open('/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # not considering data with throttle below 0.2
        #if line['throttle']>0.2:
        lines.append(line)
   
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    #current_path = '/root/Desktop/Data/IMG/' + filename
    current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/TestImages/' + filename
    image = plt.imread(current_path,format="jpg")
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/test4.jpg',images[4],format="jpg")   

def randBright(image, br=0.05):
    """Function to randomly change the brightness of an image
    
    Args: 
      image (numpy array): RGB array of input image
      br (float): V-channel will be scaled by a random between br to 1+br
    Returns:
      numpy array of brighness adjusted RGB image of same size as input
    """
    print('image rbg ',image[:,:,2])
    rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_bright = br-np.random.randint(0,50)
    print('image1 ',rand_image[:,:,2], 'random ',rand_bright)
    rand_image[:,:,1] = rand_image[:,:,1]+rand_bright
    print('image2 ',rand_image[:,:,2], 'random ',rand_bright)
    rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
    return rand_image



bright_image=randBright(images[4], br=10)

plt.imsave('/home/workspace/CarND-Behavioral-Cloning-P3/test5.jpg',bright_image,format="jpg")

num_bins = 10
n, bins, patches = plt.hist(measurements, num_bins, facecolor='blue', alpha=0.5)
plt.title("Training Dataset - Steering Angles")
plt.savefig('/home/workspace/CarND-Behavioral-Cloning-P3/Train_Dataset.png')
#plt.show()

