
from params import *

def tensor_to_predictions():
    raw_predictions = torch.load(test_out_file).detach().cpu()
    r = {}
    for i in raw_predictions:
        i = i.item()
        i = round(i/2, 1)
        if i not in r:
            r[i] = 1
        else:
            r[i] += 1
    print(r)