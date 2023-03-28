import numpy as np
import torch
from tqdm import tqdm
import time
from utils.utils import EMA, label2onehot

class Trainer:

    def __init__(self, args, model, criterion, optimizer, lr_scheduler, train_data_loader, valid_data_loader=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.save_frequency = self.args.save_frequency
        self.start_epoch = args.start_epoch

        # EMA初始化
        self.ema = EMA(0.999)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema.register(name, param.data)

    def training(self):

        for epoch in range(self.start_epoch, self.args.train_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()

            global val_loss
            if self.do_validation:
                val_loss = self.validate_epoch()
                print('Epoch:{}, learning_rate:{}, train_loss:{}, val_loss:{}, epoch_time:{}'.format(epoch, 
                                                                                               self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                                                               train_loss, 
                                                                                               val_loss, 
                                                                                               time.time() - start_time
                                                                                               ))
            else:
                print('Epoch:{}, learning_rate:{}, train_loss:{}, epoch_time:{}'.format(epoch,
                                                                                        self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                                                        train_loss, 
                                                                                        time.time() - start_time
                                                                                        ))
            if epoch != self.start_epoch and epoch % self.save_frequency == 0:
                self.model.save_checkpoint(self.args.module_save_root_path,
                                           epoch,
                                           val_loss if self.do_validation else train_loss,
                                           optimizer=self.optimizer,
                                           lr_scheduler=self.lr_scheduler)
            
    def train_epoch(self):
        self.model.train()
        mean_loss = []
        for batch_idx, (ct, target) in enumerate(tqdm(self.train_data_loader)):
            ct = ct.to(self.args.device)
            # target = label2onehot(target, self.args.classes)
            target = target.to(self.args.device)
            output1= self.model(ct)
            # print(output1.shape, target.shape)
            loss, _ = self.criterion(output1, target)
            #loss2, _ = self.criterion(output2, target)

            #loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EMA 更新
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.ema(name, param.data)

            mean_loss.append(loss.item())
        mean_loss = sum(mean_loss) / len(mean_loss)
        self.lr_scheduler.step()
        return mean_loss

    def validate_epoch(self):
        self.model.eval()
        mean_val_loss = []
        for batch_idx, (ct, target) in enumerate(self.valid_data_loader):
            with torch.no_grad():
                ct = ct.to(self.args.device)
                # target = label2onehot(target, self.args.classes)
                target = target.to(self.args.device)
                predict = self.model(ct)
                loss, _ = self.criterion(predict, target)
                #loss2, _ = self.criterion(predict2, target)
                #loss = loss1 + loss2
                mean_val_loss.append(loss.item())

        mean_val_loss = sum(mean_val_loss) / len(mean_val_loss)
        return mean_val_loss