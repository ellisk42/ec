# protonet score

import torch
import numpy as np

from PIL import Image

from torch import nn

size = 28


def load_image(array):
    array = list(map(float, map(int, array)))
    np_image = np.resize(array, (size, size))
    torch_image = torch.from_numpy(np.array(np_image, np.float32, copy=False))
    torch_image = (torch_image / 255).transpose(0, 1).contiguous()
    view_image = 1.0 - torch_image.view(1, size, size)
    return torch.unsqueeze(view_image, 0)


def load_image_path(path):
    _, _, _, a = Image.open("data/geometry/data/"+path+"/output_l.png").split()
    resized_a = a.resize((size, size), resample=Image.BILINEAR)
    np_array = np.array(resized_a, np.float32, copy=False)
    torch_image = torch.from_numpy(np.array(np_array, np.float32, copy=False))
    torch_image = (torch_image / 255).transpose(0, 1).contiguous()
    view_image = 1.0 - torch_image.view(1, size, size)
    return torch.unsqueeze(view_image, 0)


class PretrainedProtonetDistScore(nn.Module):
    def __init__(self, path):
        super(PretrainedProtonetDistScore, self).__init__()
        print("LOADING TRAINED MODEL")
        self.model = torch.load(path)
        print("LOADED TRAINED MODEL")
        # set it to not train
        self.model.requires_grad = False

    def score(self, images, targets):  # TODO make batch compatible (Max)

        xs = targets.view(-1, 1, 1, targets.size(2), targets.size(3))
        xq = images.view(-1, 1, 1, images.size(2), images.size(3))

        d = {'xs': xs, 'xq': xq}

        results = self.model.myloss(d)

        return results[1]




if __name__ == '__main__':
    model = PretrainedProtonetDistScore("./results/best_model.pt")

    i1 = "data/geometry/data/213_snake/213_snake_random_9.png"
    i2 = "data/geometry/data/213_snake/213_snake_random_8.png"
    i2 = "data/geometry/data/405_4by4squares/405_4by4squares_random_30.png"

    x = load_image_path(i1)
    y = load_image_path(i2)

    res = model.score(x, y)
    print(res['loss'])
