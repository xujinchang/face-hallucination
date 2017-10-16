import torch
from PIL import Image
import os
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable
import time
from EBGAN import generator

save_folder = './save_demo/'
#save_folder = './save_model3_vgg_gan/'

# image_list = os.listdir(image_folder)
# image_list = []
# fp = open('ball_list','r')
# for line in fp.readlines():
#      line = line.strip()
#      image_list.append(line)

G_model = generator('celebA')
G_model.load_state_dict(torch.load('./models/celebA/EBGAN/EBGAN_G_14.pkl'))
print G_model
G_model = G_model.cpu()
image_list = ['./111000.jpg']

count = 0
num = 0
for image_name in image_list:
     # image = Image.open(os.path.join(image_folder, image_name))
     # print(os.path.join(image_folder, image_name))
     print image_name
     image = Image.open(image_name)
     crop = transforms.CenterCrop(160)
     image = crop(image)
     scale = transforms.Scale(64)
     image = scale(image)
     input = Variable(ToTensor()(image)).view(1, -1, image.size[1], image.size[0])
     # input = input.cuda(device_id = 3
     #      )
     start = time.time()
     out = G_model(input)
     out = out.cpu()
     out = out.data[0]
     out[out > 1] = 1
     out[out < 0] = 0
     out_image = ToPILImage()(out)
     end = time.time()
     count += end - start
     num += 1
     save_image_name = image_name.split('/')[-1]
     out_image.save(os.path.join(save_folder, save_image_name))
     print "count:", num
print "time: ", count
