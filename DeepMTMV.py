import argparse
import os
import torch
import data_loader
from torch.utils.data import DataLoader
import numpy as np
from model import CNN_Text
import torch.nn.functional as F
import time
from sklearn.metrics import f1_score

def main(args):
    # # Device configuration
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    num_epochs = 80
    num_classes = 8
    learning_rate = 0.08
    num_views = 3
    num_layers = 4
    data_path = args.dir
    file_list = ['./data/train_web_content.npy', './data/train_web_links.npy', './data/train_web_title.npy',
                 './data/test_web_content.npy', './data/test_web_links.npy', './data/test_web_title.npy',
                 './data/train_label.npy', './data/test_label.npy']
    aaa = list(map(os.path.exists, file_list))
    if sum(aaa) != len(aaa):
        print('Raw data has not been pre-processed! Start pre-processing the raw data.')
        data_loader.preprocess(data_path)
    else:
        print('Loading the existing data set...')
    # train_dataset = data_loader.Load_datasets('train', num_classes)
    train_dataset = data_loader.Load_datasets('train', 8)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    input_dims = np.array(train_dataset.data[0]).shape
    model = CNN_Text(input_dims, [64, 32, 32, 32], [1, 2, 3, 4], num_classes, 0.5, num_layers, num_views).to(device)
    model = model.double()
    model.device = device
    model.learning_rate = learning_rate
    model.epoch = 0
    if args.model != None:
        model.load_state_dict(torch.load(args.mpodel))
        print('Successfully load pre-trained model!')
    # train the model until the model is fully trained
    train_model(model, train_loader, num_epochs)
    print('Finish training process!')
    evaluation(model)


def evaluation(model):
    test_dataset = data_loader.Load_datasets('test', 8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    device = model.device
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        prediction = []
        truth = []
        for i, sample_batched in enumerate(test_loader):
            view_1 = sample_batched['view_1'].to(device)
            view_2 = sample_batched['view_2'].to(device)
            view_3 = sample_batched['view_3'].to(device)
            labels = sample_batched['label'].to(device)
            labels = labels.long()
            outputs = model(view_1, view_2, view_3)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prediction = prediction + list(predicted.cpu().numpy())
            truth = truth + list(labels.cpu().numpy())
        print('Test Accuracy of the model on the {} test documents: {} %'.format(test_loader.dataset.label.shape[0],
                                                                                 100 * correct / total))
        print('F1 scores of the model on the {} test documents: {} %'.format(test_loader.dataset.label.shape[0],
                                100 * f1_score(truth, prediction, average='macro')))


def train_model(model, train_loader, max_epochs):
    epoch = model.epoch
    while True:
        model.learning_rate = model.learning_rate / (1 + 0.005*epoch)
        # optimizer = torch.optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        start_time = time.time()
        for i, sample_batched in enumerate(train_loader):
            view_1 = sample_batched['view_1'].to(model.device)
            view_2 = sample_batched['view_2'].to(model.device)
            view_3 = sample_batched['view_3'].to(model.device)
            labels = sample_batched['label'].to(model.device)
            optimizer.zero_grad()
            outputs = model(view_1, view_2, view_3)
            loss = F.cross_entropy(outputs, labels.long())
            loss.backward()
            optimizer.step()
            # evaluate error
            for j in range(model.num_view):
                model.scores[str(j)] = outputs
                model._update_error_data(labels, j)
        epoch += 1
        print('Epoch [{}/{}],  Loss: {:.4f}'.format(epoch, max_epochs, loss.item()))
        if epoch % 10 == 0:
            print("--- %s seconds for epoch %d---" % (time.time() - start_time, epoch))
            if model.improve_model():
                torch.save(model.state_dict(), 'model_{}.ckpt'.format(epoch))
                model.epoch = epoch
        if epoch > max_epochs:
            torch.save(model.state_dict(), 'model_{}.ckpt'.format(epoch))
            break


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='DeepMTMV')
    parser.add_argument('-n', dest='name', type=str, default='text', help='which dataset is used for demo')
    parser.add_argument('-d', dest='dir', type=str, default='C:/Users/leo/Dropbox/1dCNN/data/webkb', help='The directory of the webkb dataset')
    parser.add_argument('-g', dest='gpu', type=int, default=0, help='the index of the gpu to use')
    parser.add_argument('-m', dest='model', type=str, default=None, help='the name of the pre-trained model')
    parser.add_argument('-s', dest='num', type=int, default=8000, help='the size of training data set')
    args = parser.parse_args()
    main(args)
