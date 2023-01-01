import logging
import os

from params import *

import camembert_manager_custom as cmc
import camembert_manager_2 as cm2
import results as ft

mode = "2"

if __name__ == '__main__':

    logging.info("program is starting")
    logging.info("mode "+mode)
    if mode == "custom":
        if not os.path.exists(checkpoints_folder+"/best_model.pth"):
            cmc.full_train_with_valid(5)
        if not os.path.exists(test_out_file):
            cmc.test()
        ft.tensor_to_predictions()
    elif mode =="2":
        if not os.path.exists(checkpoints_folder+"/best_model_2.pth"):
            cm2.fully_train(2)
    else:
        logging.info(f"mode:{mode} doesnt exist")
    # valid(model, train_loader)
    logging.info("program end")
