import logging
import os

from params import *

import depreciated_camembert_manager_custom as cmc
import camembert_manager_2 as cm2
import results as ft

mode = "2"
train = True

if __name__ == '__main__':

    logging.info("program is starting")
    logging.info("mode "+mode)
    if mode == "custom":
        if not os.path.exists(checkpoints_folder+"/best_model.pth"):
            cmc.full_train_with_valid(5)
        if not os.path.exists(test_out_file):
            cmc.test()
        ft.tensor_to_predictions()
    elif mode == "2":
        if train:
            if not os.path.exists(checkpoints_folder+f"/best_model_{execution_id}.pth"):
                cm2.fully_train(2)
            else:
                cm2.fully_train(1, load=True)
        if os.path.exists(checkpoints_folder + f"/best_model_{execution_id}.pth"):
            cm2.test()
        else:
            logging.info("Error, no model")
    else:
        logging.info(f"mode:{mode} doesnt exist")
    # valid(model, train_loader)
    logging.info("program end")
