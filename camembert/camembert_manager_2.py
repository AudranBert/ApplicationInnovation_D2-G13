import pickle

import torch
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AdamW
from params import *
import torch.nn.functional as F
import custom_dataset as cd
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# https://ledatascientist.com/analyse-de-sentiments-avec-camembert/

def dataset_to_pickle_2(dataset_name, note=True):
    if not os.path.exists(os.path.join(pickle_folder, dataset_name+"_2.p")):
        logging.info(f"Loading xml of: {dataset_name}")
        df_data = pd.read_xml(os.path.join(xml_folder, dataset_name + ".xml"))
        df_data.fillna('a', inplace=True)
        os.makedirs(pickle_folder, exist_ok=True)
        logging.info(f"Tokenization of: {dataset_name}")

        reviews = df_data['commentaire'].values.tolist()
        tokenizer = CamembertTokenizer.from_pretrained(
            'camembert-base',
            do_lower_case=True)
        encoded_batch = tokenizer.batch_encode_plus(reviews,
                                                    add_special_tokens=True,
                                                    max_length=512,
                                                    padding=True,
                                                    truncation=True,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        if note:
            df_token = pd.DataFrame(columns=['commentaire', 'note'])
            df_token['note'] = df_data['note'].apply(to_float)
            sentiments = df_token['note'].values.tolist()
            sentiments = torch.tensor(sentiments)
            dataset = TensorDataset(
                encoded_batch['input_ids'],
                encoded_batch['attention_mask'],
                sentiments)
        else:
            dataset = TensorDataset(
                encoded_batch['input_ids'],
                encoded_batch['attention_mask'])
        with open(os.path.join(pickle_folder, dataset_name+"_2.p"), 'wb') as f:
            pickle.dump(dataset, f)
        logging.info(f"Saving pickle: {dataset_name}")
    else:
        logging.info(f"Loading pickle: {dataset_name}")
        with open(os.path.join(pickle_folder, dataset_name+"_2.p"), 'rb') as f:
            dataset = pickle.load(f)
    # df_token['commentaire'] = df_data['commentaire']
    # print(df_token)
    return dataset


def init_training():
    train_dataset = dataset_to_pickle_2("train")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=4,
        persistent_workers=True
    )

    valid_dataset = dataset_to_pickle_2("dev")
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=64,
    )

    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels=10).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5, # Learning Rate
                      eps=1e-8 # Epsilon
    )
    return model, optimizer, train_loader, valid_loader

def acc_10_c(out, target):
    t = target
    pred = torch.argmax(out, dim=1).to(device).type(torch.float)
    return pred.eq(t.view_as(pred)).sum()

def acc_3_c(out, target):
    t = target
    t = torch.where(t < 5, 0, t)
    t = torch.where(t == 5, 1, t)
    t = torch.where(t > 5, 2, t)
    pred = torch.argmax(out, dim=1).to(device).type(torch.float)
    pred = torch.where(pred < 5, 0, t)
    pred = torch.where(pred == 5, 1, t)
    pred = torch.where(pred > 5, 2, t)
    return pred.eq(t.view_as(pred)).sum()

def valid(epoch, model, valid_loader, during_epoch=False):
    model.eval()
    correct_10_tot = 0
    correct_3_tot = 0
    with torch.no_grad():
        for batch_idx, (x, mask, target) in enumerate(tqdm(valid_loader, desc=str(epoch) + ": Valid batch")):  # for each batch
            x, attention_mask, target = x.to(device), mask.to(device), target.to(device)
            out = model(x, attention_mask=attention_mask)
            out = out[0]
            correct_10_tot += acc_10_c(out, target)
            correct_3_tot += acc_3_c(out, target)
            break
    v_10_acc = correct_10_tot / len(valid_loader.dataset)
    v_3_acc = correct_3_tot / len(valid_loader.dataset)
    if during_epoch:
        writer.add_scalar('Accuracy_10c/valid_step', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid_step', v_3_acc, epoch)
    else:
        writer.add_scalar('Accuracy_10c/valid', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid', v_3_acc, epoch)
    return v_10_acc


def fully_train(nb_epoch):
    os.makedirs(checkpoints_folder, exist_ok=True)
    model, optimizer, train_loader, valid_loader = init_training()
    best_valid_acc = 0
    v_acc = valid(get_step(0, train_loader, 0), model, valid_loader, during_epoch=True)
    if v_acc > best_valid_acc:  # keep the best weights
        best_valid_acc = v_acc
        torch.save(model, checkpoints_folder + "/best_model_2.pth")

    for epoch in range(1, nb_epoch+1):
        total_train_loss = 0
        t_loss = 0
        t_batch = 0
        model.train()

        # Pour chaque batch
        for batch_idx, (x, mask, target) in enumerate(tqdm(train_loader, desc=str(epoch) + ": Train batch")):  # for each batch
            x, attention_mask, target = x.to(device), mask.to(device), target.to(torch.int64).to(device)  # send it to the device

            model.zero_grad()
            loss, logits = model(x,
                                 token_type_ids=None,
                                 attention_mask=attention_mask,
                                 labels=target,
                                 return_dict=False)

            total_train_loss += loss.item()
            t_loss += loss.item()
            t_batch += 1
            loss.backward()
            optimizer.step()

            if get_step(epoch, train_loader, batch_idx) % 1000 == 0 and batch_idx != 0:
                size = len(target) * t_batch
                step_loss = t_loss / size
                writer.add_scalar('Loss/train', step_loss, get_step(epoch, train_loader, batch_idx))
                t_loss = 0
                t_batch = 0
            if get_step(epoch, train_loader, batch_idx) % 25000 == 0 and batch_idx != 0:
                torch.save(model, checkpoints_folder + "/last_model_2.pth")
                v_acc = valid(get_step(epoch, train_loader, batch_idx), model, valid_loader, during_epoch=True)
                if v_acc > best_valid_acc:  # keep the best weights
                    best_valid_acc = v_acc
                    torch.save(model, checkpoints_folder + "/best_model_2.pth")
        # On calcule la  loss moyenne sur toute l'epoque
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        logging.info(f"Avg train loss :{avg_train_loss}")
        torch.save(model, checkpoints_folder + "/last_model_2.pth")
        v_acc = valid(get_step(epoch, train_loader, batch_idx), model, valid_loader, during_epoch=True)
        if v_acc > best_valid_acc:  # keep the best weights
            best_valid_acc = v_acc
            torch.save(model, checkpoints_folder + "/best_model_2.pth")