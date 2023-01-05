from torch.utils.data import  DataLoader
from tqdm import tqdm
from transformers import CamembertForSequenceClassification, AdamW
from params import *
from data_prep_2 import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# https://ledatascientist.com/analyse-de-sentiments-avec-camembert/




def init_training(load=False):
    train_dataset = dataset_to_pickle_2("train")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=4,
        # persistent_workers=True
    )

    valid_dataset = dataset_to_pickle_2("dev")
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=64,
    )

    if load:
        logging.info("Load a checkpoint")
        model = torch.load(checkpoints_folder+f"/last_model_{execution_id}.pth").to(device)
    else:
        logging.info("Load a pretrained model")
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
    pred = torch.argmax(out, dim=1).to(device).type(torch.int64)
    return pred.eq(t.view_as(pred)).sum()

def acc_3_c(out, target):
    t = target.type(torch.int64)
    t = torch.where(t < 5, 0, t)
    t = torch.where(t == 5, 1, t)
    t = torch.where(t > 5, 2, t)
    pred = torch.argmax(out, dim=1).to(device).type(torch.int64)
    pred = torch.where(pred < 5, 0, pred)
    pred = torch.where(pred == 5, 1, pred)
    pred = torch.where(pred > 5, 2, pred)
    return pred.eq(t.view_as(pred)).sum()

def test():
    os.makedirs(export_folder, exist_ok=True)
    test_dataset = dataset_to_pickle_2("test", note=False)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=64,
    )
    model = torch.load(checkpoints_folder + f"/best_model_{execution_id}.pth").to(device)
    model.eval()
    model_predictions = []
    with torch.no_grad():
            for batch_idx, (x, mask) in enumerate(tqdm(test_loader, desc="Test batch")):  # for each batch
                x, attention_mask = x.to(device), mask.to(device)
                out = model(x, attention_mask=attention_mask)
                out = out[0]
                pred = torch.argmax(out, dim=1).cpu().detach().type(torch.float)
                model_predictions.append(pred)
    predictions = torch.cat(model_predictions, dim=0)
    torch.save(predictions, os.path.join(export_folder, f"{test_out_file}_{execution_id}.pth"))

def valid(epoch, model, valid_loader, during_epoch=False, both=False):
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
    v_10_acc = correct_10_tot / len(valid_loader.dataset)
    v_3_acc = correct_3_tot / len(valid_loader.dataset)
    if during_epoch:
        writer.add_scalar('Accuracy_10c/valid_step', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid_step', v_3_acc, epoch)
    elif both:
        writer.add_scalar('Accuracy_10c/valid_step', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid_step', v_3_acc, epoch)
        writer.add_scalar('Accuracy_10c/valid', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid', v_3_acc, epoch)
    else:
        writer.add_scalar('Accuracy_10c/valid', v_10_acc, epoch)
        writer.add_scalar('Accuracy_3c/valid', v_3_acc, epoch)
    return v_10_acc


def fully_train(nb_epoch, load=False, only_init=False):
    os.makedirs(checkpoints_folder, exist_ok=True)
    model, optimizer, train_loader, valid_loader = init_training(load)
    if only_init:
        return
    best_valid_acc = 0
    v_acc = valid(0, model, valid_loader, both=True)
    if v_acc > best_valid_acc:  # keep the best weights
        best_valid_acc = v_acc

    for epoch in range(1, nb_epoch+1):
        total_train_loss = 0
        b_loss = 0
        b_ct = 0
        b_acc3 = 0
        b_acc10 = 0
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
            b_acc3 += acc_3_c(logits, target)
            b_acc10 += acc_10_c(logits, target)
            total_train_loss += loss.item()
            b_loss += loss.item()
            b_ct += 1
            loss.backward()
            optimizer.step()

            if get_step(epoch, train_loader, batch_idx) % 1000 == 0 and batch_idx != 0:
                size = len(target) * b_ct
                writer.add_scalar('Loss/train', (b_loss / size), get_step(epoch, train_loader, batch_idx))
                writer.add_scalar('Accuracy_3c/train', (b_acc3 / size)*100, get_step(epoch, train_loader, batch_idx))
                writer.add_scalar('Accuracy_10c/train', (b_acc10 / size)*100, get_step(epoch, train_loader, batch_idx))
                b_loss = 0
                b_ct = 0
                b_acc3 = 0
                b_acc10 = 0
            if get_step(epoch, train_loader, batch_idx) % 25000 == 0 and batch_idx != 0:
                torch.save(model, checkpoints_folder + f"/last_model_{execution_id}.pth")
                v_acc = valid(get_step(epoch, train_loader, batch_idx), model, valid_loader, during_epoch=True)
                if v_acc > best_valid_acc:  # keep the best weights
                    best_valid_acc = v_acc
                    torch.save(model, checkpoints_folder + f"/best_model_{execution_id}.pth")
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        logging.info(f"Avg train loss :{avg_train_loss}")
        torch.save(model, checkpoints_folder + f"/last_model_{execution_id}.pth")
        v_acc = valid(get_step(epoch, train_loader, batch_idx), model, valid_loader, during_epoch=True, both=True)
        if v_acc > best_valid_acc:  # keep the best weights
            best_valid_acc = v_acc
            torch.save(model, checkpoints_folder + f"/best_model_{execution_id}.pth")