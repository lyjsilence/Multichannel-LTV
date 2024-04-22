import os
import time
import numpy as np
import pandas as pd
import torch
import utils


def Trainer(model, dl_train, dl_val, dl_test, args, device, exp_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.0001)
    min_loss = np.inf

    if args.model_name == 'single_ltv':
        save_path = os.path.join(args.save_dirs, args.model_name, str(args.company_id), 'exp_' + str(exp_id))
    else:
        save_path = os.path.join(args.save_dirs, args.model_name, 'exp_' + str(exp_id))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.log:
        df_log_val = pd.DataFrame()

    for epoch in range(1, args.epoch_max + 1):
        '''Model training'''
        model.train()
        epoch_start_time = time.time()

        training_class_loss, training_reg_loss = [], []

        for i, batch in enumerate(dl_train):
            x_cat = batch["x_cat"]
            x_num = batch["x_num"]
            y = batch["y"]

            # get the model estimation
            x = model(x_cat, x_num)

            # compute the ZILN loss
            class_loss, reg_loss = model.compute_loss(x, y)

            loss = class_loss + reg_loss

            training_class_loss.append(class_loss.item())
            training_reg_loss.append(reg_loss.item())

            print('\rEpoch [{}/{}], Batch [{}/{}], Class Loss: {:.4f}, Reg Loss: {:.4f}, time elapsed: {:.2f}, '
                  .format(epoch, args.epoch_max, i + 1, len(dl_train), np.mean(training_class_loss), np.mean(training_reg_loss),
                          time.time() - epoch_start_time), end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        time_elapsed = time.time() - epoch_start_time

        if args.log:
            df_log_val.loc[epoch, 'epoch'] = epoch
            df_log_val.loc[epoch, 'time elapsed'] = time_elapsed
            df_log_val.loc[epoch, 'train_class_Loss'] = np.mean(training_class_loss)
            df_log_val.loc[epoch, 'train_reg_Loss'] = np.mean(training_reg_loss)


        '''Modeling val'''
        with torch.no_grad():
            model.eval()
            val_class_loss, val_reg_loss = [], []

            for i, batch in enumerate(dl_val):
                x_cat = batch["x_cat"]
                x_num = batch["x_num"]
                y = batch["y"]

                x = model(x_cat, x_num)

                # compute the ZILN loss
                class_loss, reg_loss = model.compute_loss(x, y)
                val_class_loss.append(class_loss.item())
                val_reg_loss.append(reg_loss.item())

            val_class_loss = np.mean(val_class_loss)
            val_reg_loss = np.mean(val_reg_loss)

            print('Val, Class Loss: {:.4f}, Reg Loss: {:.4f}:'.format(val_class_loss, val_reg_loss))

            # save the best model parameter
            if val_class_loss + val_reg_loss < min_loss:
                min_loss = val_class_loss + val_reg_loss

                model_save_path = os.path.join(save_path, args.model_name + '.pkl')
                torch.save(model.state_dict(), model_save_path)

            if args.log:
                df_log_val.loc[epoch, 'val_class_loss'] = np.mean(val_class_loss)
                df_log_val.loc[epoch, 'val_reg_loss'] = np.mean(val_reg_loss)

        scheduler.step(val_class_loss + val_reg_loss)

    '''Modeling test'''
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))

        for i, batch in enumerate(dl_test):
            x_cat = batch["x_cat"]
            x_num = batch["x_num"]
            y = batch["y"]

            x = model(x_cat, x_num)

            # compute the AUC and normalized GINI coefficient
            aucroc = utils.compute_metrics(x, y, model_name=args.model_name)
            y_pred = model.predict(x)
            gini = utils.compute_gini(y_pred, y, model_name=args.model_name)

        res = pd.DataFrame({'AUCROC': aucroc, 'GINI': gini})
        res.to_csv(os.path.join(save_path, 'results.csv'))

    if args.log:
        val_log_save_path_test = os.path.join(save_path, args.model_name + '_val_log.csv')
        df_log_val.to_csv(val_log_save_path_test)

