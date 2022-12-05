#from analyse_dataset.load import *
import pandas as pd
import numpy as np
from main_svm import *

def createEvalFile(filename_eval, dataset, liblinear_output_file):
    liblinear_out = pd.read_csv(liblinear_output_file, sep=" ", header=None,names=["note"])
    out = pd.concat([dataset["review_id"], liblinear_out], axis=1)
    print(out)
    out["note"] = pd.to_numeric(out["note"],downcast="float")
    out.to_csv(filename_eval, header=None, index=None, sep=' ',decimal=",")

if __name__ == '__main__':
    pickle_file = os.path.join(pickle_folder, "test.p")
    test = check_xml(pickle_file, test_file)
    createEvalFile("eval.txt", test, "out.txt")
#df_dev = load_xml("dataset/dev.xml")
#save_object(df_dev,os.path.join("pickle","df_dev.pickle"))
#liblinear_out = pd.read_csv('out.txt', sep=" ",header=None)
#out = pd.concat([df_dev["review_id"],liblinear_out],axis=1)
#out.to_csv("out_eval_panda.txt",header=None,index=None, sep=' ')