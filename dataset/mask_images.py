from PIL import Image
import numpy as np
import torch
def mask_img(img_path):
    img_pil=Image.open(img_path)
    img_pil_arr = np.array(img_pil)
    img_tensor = torch.tensor(img_pil_arr)
    img_for_mask = img_tensor.permute(2,0,1)
    mask_position = [742,2469,206,51]
    zeros = torch.zeros_like(img_for_mask)
    zeros[:,206:742,51:2469]=1
    img_for_mask.masked_fill_(torch.ByteTensor(zeros),value=torch.tensor(0))
    inverse_img = img_for_mask.permute(1,2,0)
    masked_img = inverse_img.numpy()
