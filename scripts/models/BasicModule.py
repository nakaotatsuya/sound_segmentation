import os
import torch
import torch.nn as nn
import time

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, save_dir):
        #filename = time.strftime('%m%d_' + self.model_name + '.pth')
        filename = time.strftime(self.model_name + '.pth')
        torch.save(self.state_dict(), os.path.join(save_dir, filename))

# if __name__ == "__main__":
#     bm = BasicModule()
#     print(bm.model_name)
#     print(time.strftime(bm.model_name + ".pth"))
