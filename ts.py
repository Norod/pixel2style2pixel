import torch
import PIL
from PIL import Image, ImageOps
import numpy as np

def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

def load_image_as_array(path):
    image_in = path
    im = Image.open(image_in)
    try:
        im = ImageOps.exif_transpose(im)
    except:
        print("exif problem, not rotating")
        im = im.convert("RGB")

    im = im.resize((256, 256))
    im_array = np.array(im, np.float32)
    im_array = (im_array/255)*2 - 1
    im_array = np.transpose(im_array, (2, 0, 1))
    im_array = np.expand_dims(im_array, 0)

    return im_array

def run():
    im_array = load_image_as_array('test_data/face-ok.jpg')
    tensor_in = torch.Tensor(im_array)

    test_image = tensor2im(tensor_in[0])
    #test_image.show()

    net = torch.jit.load('p2s2p_torchscript.pt')
    net.eval()
    net.cuda()
    #result = net(tensor_in)
    tensor_in = tensor_in.cuda().float()
    traced_model = torch.jit.trace(net, tensor_in)
    result = traced_model(tensor_in)

    result = result.cpu().float()

    output_image = tensor2im(result[0])
    output_image.save('face-toon.jpg')
    #output_image.show()

if __name__ == '__main__':
	run()
