import os
import cv2 
# import matplotlib.pyplot as plt


class imageLabel:
	def __init__(self, images_dir, labels_dir):
		self.images_dir, self.labels_dir = images_dir, labels_dir
		self.find_unlabeled_images()

	def find_unlabled_images(self):


def draw_circle(event,x,y,flags,param):
	global mouseX,mouseY, image
	if event == cv2.EVENT_LBUTTONDBLCLK:
		print('x = {}, y = {}'.format(x, y))
		image_updated = cv2.circle(image, (x,y), 6, (255,0,0), -1)
		cv2.imshow('image', image_updated)
		mouseX,mouseY = x,y



def do_the_labeling(images_dir, labels_dir):
	cv2.namedWindow('image')
	cv2.setMouseCallback('image',draw_circle)
	cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	global image

	all_images = [os.path.join(images_dir, img) for img in os.listdir(images_dir)]
	for image_name in all_images:
		print(image_name)
		image = cv2.imread(image_name)
		image = cv2.resize(image, (1920, 1080), interpolation = cv2.INTER_AREA)
		print(image.shape)
		
		cv2.imshow('image', image)
		k = cv2.waitKey(0)
		if k == ord('q'):
			break
		elif k == ord('a'):
			print(mouseX, mouseY)
	


		



def main():
	base_dir = '/home/majd/drone_racing_ws/catkin_ddr/src/basic_rl_agent/data/'
	images_dir = os.path.join(base_dir, 'real_gate_data/images')
	labels_dir = os.path.join(base_dir, 'real_gate_data/labels')
	do_the_labeling(images_dir, labels_dir)







if __name__ == '__main__':
	main()
