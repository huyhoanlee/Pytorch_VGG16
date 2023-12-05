import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

class BaseTransform(): #class preprocessing ảnh thành ảnh 224x224
  def __init__(self, resize, mean, std):
    self.base_transform = transforms.Compose([transforms.Resize(resize),
                                              transforms.CenterCrop(resize),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
  def __call__(self, img):
    return self.base_transform(img)
  
class Predictor():  #class dự đoán 
  def __init__(self, class_index):
    self.class_index = class_index
  def predict_max(self, out):
    maxid = np.argmax(out.detach().numpy())
    predicted_label_name = self.class_index[str(maxid)]
    return predicted_label_name

#load file json chứa tên các object
import json
class_index = json.load(open('imagenet_class_index.json',  "r"))

#load pre-trained model VGG16
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained )
net.eval()
# print(net)

#load predictor 
predictor = Predictor(class_index)

#load ảnh 
img = Image.open('dog.jpg')

resize = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = BaseTransform(resize, mean, std) #tiền xử lý ảnh 
img_transformed = transform(img)
img_transformed = img_transformed.unsqueeze_(0) 
#img_transformed.shape

#Tiến hành dự đoán
out = net(img_transformed)
result = predictor.predict_max(out)
print(result)