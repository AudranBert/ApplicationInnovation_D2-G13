import csv
import os

from params import *

def export_test_results():
    raw_predictions = torch.load(test_out_file).detach().cpu()
    f = open("test_results.csv", 'w')
    writer = csv.writer(f, lineterminator='\n')
    for i in raw_predictions:
        i = i.item()
        i += 1
        # i = round(i / 2, 1)
        writer.writerow([i])
    f.close()

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

def divide_twice(v):
    v = v/2
    return v

def createEvalFile(filename_eval, dataset, model_output_file):
    liblinear_out = pd.read_csv(model_output_file, sep=" ", header=None,names=["note"])
    out = pd.concat([dataset["review_id"], liblinear_out], axis=1)
    print(out)
    out["note"] = pd.to_numeric(out["note"],downcast="float")
    out["note"] = out["note"].apply(divide_twice)
    out.to_csv(filename_eval, header=None, index=None, sep=' ', decimal=",")



if __name__ == '__main__':
    # if not os.path.exists("test_results.csv"):
    export_test_results()
    pickle_file = os.path.join(pickle_folder, "test_set.p")
    test = check_xml(pickle_file, test_file)
    createEvalFile("eval.txt", test, "test_results.csv")