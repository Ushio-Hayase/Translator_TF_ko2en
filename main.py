import os
import time

import torch
import numpy as np
import tqdm
from data import Dataset
from model import Transformer

BATCH_SIZE = 64
EPOCHS = 10
PAD = 0

def acc_fn(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.eq(y_pred, y_true).sum().item()/len(y_pred)
    return acc

if __name__ == "__main__":

    model = Transformer(256 , 512, 119547).cuda()

    loss_object = torch.nn.CrossEntropyLoss(reduction="none")
    learning_rate = 0.01

    optimizer = torch.optim.Adam(model.parameters(),learning_rate, betas=(0.9, 0.98),
                                        eps=1e-9)


    x_train_df = torch.from_numpy(np.load("x_train.npy"))
    y_train_df = torch.from_numpy(np.load("y_train.npy"))
    x_valid_df = torch.from_numpy(np.load("x_valid.npy"))
    y_valid_df = torch.from_numpy(np.load("y_valid.npy"))


    train_dataset = Dataset(x_train_df, y_train_df, BATCH_SIZE)
    valid_dataset = Dataset(x_valid_df, y_valid_df, BATCH_SIZE)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for (enc_in, tar) in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                enc_in, tar = enc_in.cuda(), tar.cuda()

                optimizer.zero_grad()

                pad = torch.tensor([[PAD]])
                pad = pad.repeat((BATCH_SIZE, 1))

                dec_in = tar[:, :-1]
                tar_real = tar[:, 1:]

                # dec_in = torch.cat([dec_in, pad], dim=-1)
                # tar_real = torch.cat([pad, tar_real], dim=-1)

                out = model(enc_in, dec_in)
                loss = loss_object(out, tar_real)
                loss.backward()
                optimizer.step()
                
                acc = acc_fn(out, tar_real)


                tepoch.set_postfix(loss=loss.item(), acc=acc)

        with torch.no_grad():
            with tqdm.tqdm(valid_dataloader, unit="batch") as tepoch:
                for (enc_in, tar) in tepoch:
                    tepoch.set_description(f"Valid : Epoch {epoch+1}")

                    enc_in, tar = enc_in.cuda(), tar.cuda()

                    optimizer.zero_grad()

                    pad = torch.tensor([[PAD]])
                    pad = pad.repeat((BATCH_SIZE, 1))

                    dec_in = tar[:, :-1]
                    tar_real = tar[:, 1:]

                    # dec_in = torch.cat([dec_in, pad], dim=-1)
                    # tar_real = torch.cat([pad, tar_real], dim=-1)

                    out = model(enc_in, dec_in)
                    loss = loss_object(out, tar_real)
                    loss.backward()
                    optimizer.step()
                    
                    acc = acc_fn(out, tar_real)


                    tepoch.set_postfix(val_loss=loss.item(), val_acc=acc)
            

