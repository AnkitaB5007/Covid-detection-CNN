# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import datetime

import Tester as tester


import Logger as logg
import Metrics as met
import Model_utilities as mtils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def list_toTorch(list):
    return torch.from_numpy(np.array(list))

def load_model(path, nets, num_classes):
    print('[INFO] loading trained models...')
    model_list = []
    model_name = []

    model_0 = models.resnext50_32x4d(pretrained=True)
    model_0.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_0.fc.in_features, num_classes))

    model_1 = models.inception_v3(pretrained=True, aux_logits=False)
    # This already had default drop out
    model_1.fc = nn.Linear(model_1.fc.in_features, num_classes)

    model_2 = models.densenet161(pretrained=True)
    # Do we need drop out here?
    model_2.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model_2.classifier.in_features, num_classes))
    model_2 = torch.nn.DataParallel(model_2).module

    for m in nets:
        if 'densenet.pth' in m:
            model_2.load_state_dict(torch.load(m, map_location=device))
            model_name.append('DenseNet-161')
            print(m, 'Model is loaded')
        elif 'inception.pth' in m:
            model_1.load_state_dict(torch.load(m, map_location=device))
            model_name.append('Inception_v3')
            print(m, 'Model is loaded')
        elif 'resnext' in m:
            model_0.load_state_dict(torch.load(m, map_location=device))
            model_name.append('ResNeXt-50')
            print(m, 'Model is loaded')

    model_list.extend([model_0, model_1, model_2])

    return model_list, model_name

# https://mlwave.com/kaggle-ensembling-guide/
# https://www.mm218.dev/posts/2021/01/model-averaging/

def predict_with_ensemble(outs, labels):
    tot = np.zeros((len(outs[0]), outs[0][0].size))

    # Model averaging
    for matrix in outs:
        tot += matrix

    tot /= len(outs)
    
    # get argmax from each model
    _, preds = torch.max(torch.from_numpy(tot), 1)
    total_correct = torch.sum(preds == labels)

    return preds, labels, total_correct



# -------------------------- ENSEMBLE TESTING --------------------------------- #
'''
load the test loaders , test set
load the model
call the predict_with_ensemble function here
calculate the metrics
'''
def main():
    workspace = os.path.abspath("../")
    #workspace = '/content/drive/MyDrive/HLCV/Covid'
    dataset = 'Datasets'
    data_dir = os.path.join(workspace, dataset)

    num_classes = 3
    trained = 'last'
    test_dir = os.path.join(data_dir, 'Test')

    test_set_0 = datasets.ImageFolder(test_dir, mtils.create_transform('ResNeXt50'))
    test_set_1 = datasets.ImageFolder(test_dir, mtils.create_transform('Inception_v3'))

    test_size = len(test_set_0)
    classes = test_set_0.classes
    print(classes)

    test_loader0 = torch.utils.data.DataLoader(test_set_0, batch_size=1, shuffle=False, num_workers=0)
    test_loader1 = torch.utils.data.DataLoader(test_set_1, batch_size=1, shuffle=False, num_workers=0)

    loaders = [test_loader0, test_loader1, test_loader0]

    model_path = os.path.join(workspace, 'checkpoints', trained)
    nets = [os.path.join(model_path, x) for x in os.listdir(model_path)]

    model_list, model_name = load_model(model_path, nets, num_classes)
    

    test_accs, test_f1s, cms, outs = [], [], [], []

    softmax = nn.Softmax(dim=0 ) 

    for i in range(len(model_list)):
        test_acc, f1_test, cm, out, labels = tester.test(model_list[i], device, loaders[i], test_size, model_name[i])

        out = [softmax(x).cpu().numpy() for x in out]
        out = np.asmatrix(out)

        test_accs.append(test_acc)
        test_f1s.append(f1_test)
        cms.append(cm)
        outs.append(out)

        met.plot_confusion_matrix(cm, classes, model_name[i] + '_' + str(num_classes), workspace, model_name[i] + ' - Acc: ' + str(round(test_acc.item(), 3)) + '%', save=False)
        logg.create_test_log(workspace, cm, test_acc, f1_test, model_name[i])

    start = datetime.datetime.now()
    preds, labels, total_correct = predict_with_ensemble(outs, list_toTorch(labels))
    end = datetime.datetime.now()
    elapsed = end - start

    # calculate here accuracu and f1-score
    total_acc = total_correct.numpy() / len(preds.numpy())
    total_fscore = f1_score(labels, preds, average='micro')
    total_cm = confusion_matrix(labels, preds)

    print('\n[INFO] ensemble model testing complete')
    print('- total accuracy = ', total_acc)
    print('- total F1-score = ', total_fscore)
    print('- elapsed time (microsec) = ', elapsed.microseconds)
    
    # does not work, list index error???
    #met.compute_AUC_scores(labels, preds, classes)

    #timestamp = str(datetime.datetime.now()).split('.')[0]
    met.plot_confusion_matrix(total_cm, classes, 'Ensemble_approach' + str(num_classes) + '_' + trained, workspace, 'Ensemble - acc: ' + str(round(total_acc.item(),3)) + '%', save=False)


if __name__ == "__main__":
    main()
