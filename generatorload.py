import pickle
import numpy as np 
import tensorflow as tf 
import PIL.Image
from matplotlib import cm
import misc
import cv2
import os
import imageio

tf.InteractiveSession()
def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img=imageio.imread(str(os.path.join(folder,filename)))
        if img is not None:
            images.append(img)
    return images
folder="/root/UNOSAT/validation"
filenames = os.listdir(folder)

images = load_images_from_folder(folder)

#image_gen=imageio.imread(str("/root/UNOSAT/validation/tile_11_43.png"))

with open('network-snapshot-006000.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

Gs.print_layers()

print(Gs.input_shapes)
count=0
for img in images:
    M = img.shape[0]//2
    N = img.shape[1]//2
    tiles = [img[x:x+M, y:y+N] for x in range(0, img.shape[0], M) for y in range(0, img.shape[1], N)]
    real1 = tiles[0].reshape(1,1,128,128)
    real1 = (real1.astype(np.float32)-127.5)/127.5
    real2 = tiles[1].reshape(1,1,128,128)
    real2 = (real2.astype(np.float32)-127.5)/127.5
    real3 = tiles[2].reshape(1,1,128,128)
    real3 = (real3.astype(np.float32)-127.5)/127.5
    
    drange_net=[-1,1]
    rnd = np.random.RandomState(55)
    #latents = rnd.randn(1, Gs.input_shapes[0])
    latents = rnd.randn(1, 1, 128, 128)
    left = np.concatenate((real1, real3), axis=2)
    right = np.concatenate((real2, latents), axis=2)
    lat_and_cond = np.concatenate((left, right), axis=3)
    #latents = rnd.randn(1, *Gs.input_shapes[0][1:])
    #print(latents.shape)
    #latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]
    labels=np.zeros((1,0))
    #labels = np.zeros([latents.shape[0]] + Gs.input_shapes[0][1:])
    #print(labels.shape)
    
    gen_images = Gs.run(lat_and_cond, labels)
    #print(images.shape)
    
    #images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
    #images = images.transpose(0,2,3,1)
    #print(images.shape)
    #print(images[0].shape)
    #print(images[0].shape)
    #print(tiles[0].shape)
    
    #reshaped_im=images[0].reshape((128,128))
    #print(reshaped_im.shape)
    def get_concat_h(im1, im2):
        dst = PIL.Image.new('L', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0,0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def get_concat_v(im1, im2):
        dst = PIL.Image.new('L', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0,0))
        dst.paste(im2, (0, im1.height))
        return dst
    #tiles_left = cv2.vconcat([tiles[0], tiles[2]])
    #tiles_right = cv2.vconcat([tiles[1], reshaped_im)
    #final_im = cv2.hconcat([tiles_left, tiles_right])
    #cv2.imwrite('image_big.png', final_im)

    #right_org = get_concat_v(tiles_lt, tiles_lb)
    #left_org = get_concat_v(tiles_rt, reshaped_im)
    #get_concat_h(right_org, left_org).save('large_image.png')
    #misc.save_image_grid(images, 'img.png', drange=drange_net, grid_size= (15,8))
    misc.save_image(gen_images[0], '/root/UNOSAT/valresults/'+filenames[count], drange=drange_net, quality=95)
    count +=1
#PIL.Image.fromarray(images[0]).save('img.png')

#for idx in range(images.shape[0]):
#    PIL.Image.fromarray(images[idx]).save('img.png')

