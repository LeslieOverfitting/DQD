import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np  

from utils import get_time_diff


def train_model(train_data_loader, test_data_loader, model, config):
    start_time = time.time()
    optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)
    model.train()
    total_batch = 0
    dev_best_loss = float('inf')
    for epoch in range(config.epochs_num):
        print('epoch[{}/{}] '.format(epoch+1, config.epochs_num))
        for (inputs, labels) in train_data_loader:
            total_batch += 1
            model.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 2000 == 0:
                true_label = labels.data.cpu()
                predict = torch.max(outputs, dim=1)[1].cpu().numpy()
                train_acc = metrics.accuracy_score(true_label, predict)
                dev_acc, dev_loss = evaluate(test_data_loader, model, flag=False)
                if dev_loss < dev_best_loss:
                    improve = '*'
                else:
                    improve = ' '
                time_diff = get_time_diff(start_time)
                msg = 'Iter: {0:>6}, Train loss: {1:>5.3}, Train acc: {2:6.2%}, Dev loss {3: 5.3}, Dev acc: {4: 6.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_diff, improve))
                model.train()
    test_model(test_data_loader, model, config)


def evaluate(test_data_loader, model, flag=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for inputs, labels in test_data_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss.time()
            predict = torch.max(outputs, dim=1)[1].cpu().numpy()
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all,labels)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if flag:
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(predict_all), report, confusion
    else:
        return acc,  loss_total / len(predict_all)

def test_model(test_data_loader, model, config):
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(test_data_loader, model, config, test=True)
    msg = "Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_diff = get_time_diff(start_time)
    print("Time usage:", time_diff)
            
