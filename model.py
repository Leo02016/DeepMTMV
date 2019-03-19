import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
from numpy import linalg as LA
from sklearn.cluster import KMeans
from easydict import EasyDict as edict

cfg = edict()
cfg.TRAIN = edict()
cfg.TRAIN.CLUSTER_REP = 10


class CNN_Text(nn.Module):
    def __init__(self, input_dim, kernel_num, kernel_sizes, num_class, dropout, num_layers, num_view):
        super(CNN_Text, self).__init__()
        (V, D1, D2) = input_dim
        C = num_class
        ks = kernel_sizes
        self.num_layers = num_layers
        self.cur_layer = num_layers
        self.convs1 = nn.ModuleList([nn.Conv2d(1, kernel_num[k], (ks[k], D2)) for k in range(len(ks))])
        self.convs2 = nn.ModuleList([nn.Conv2d(1, kernel_num[k], (ks[k], D2)) for k in range(len(ks))])
        self.convs3 = nn.ModuleList([nn.Conv2d(1, kernel_num[k], (ks[k], D2)) for k in range(len(ks))])
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.BatchNorm1d(sum(kernel_num))
        self.fc1 = nn.ModuleList([Linear(sum(kernel_num)*3, int(sum(kernel_num) / 2)*3)])
        self.fc1.indices_list = [[0]]
        self.fc1.group = False
        self.fc2 = nn.ModuleList([Linear(int(sum(kernel_num) / 2)*3, int(sum(kernel_num) / 4)*3)])
        self.fc2.indices_list = [[0]]
        self.fc2.group = False
        self.fc3 = nn.ModuleList([Linear(int(sum(kernel_num) / 4)*3, 1) for _ in range(C)])
        self.fc3.group = False
        self.fc3.indices_list = [[i for i in range(num_class)]]
        self.layer_list = ['convs', 'fc1', 'fc2', 'fc3']
        self.joint_diff_prob = dict()
        self.joint_easy_prob = dict()
        self.margin_mean = dict()
        self.err_mean = dict()
        self.err_corr = dict()
        self.cur_err_vec = dict()
        self.scores = dict()
        self.cur_round = 1
        self.epoch = 1
        self.num_view = num_view
        self.num_class = num_class
        self.err_div_factor = 1e-16
        self.error_decay_factor = 0.99
        for i in range(num_view):
            self.init_params(num_class, i)
        self.net_graph = [[] for _ in range(num_layers)]
        for i in range(num_layers-1):
            self.net_graph[i] = [[j for j in range(num_class)]]
        self.net_graph[num_layers - 1] = [[j] for j in range(num_class)]
        self.col_cost_at = [0.5/math.pow(2, i) for i in range(num_layers)]

    def init_params(self, num_class, i):
        i = str(i)
        self.joint_diff_prob[i] = np.zeros((num_class, num_class), dtype=np.float32)
        self.joint_easy_prob[i] = np.zeros((num_class, num_class), dtype=np.float32)
        self.margin_mean[i] = np.zeros((num_class,), dtype=np.float32)
        self.err_mean[i] = np.zeros((num_class,), dtype=np.float32)
        self.err_corr[i] = np.zeros((num_class, num_class), dtype=np.float32)
        self.cur_err_vec[i] = []
        self.scores[i] = []

    def oneHotEncode(self, labels):
        if self.device.type == 'cuda':
            labels = labels.cpu().numpy()
            y = np.array([np.array([1 if j == labels[i] else 0 for j in range(self.num_class)]) for i in range(len(labels))])
        else:
            y = np.array([np.array([1 if j == labels.numpy()[i] else 0 for j in range(self.num_class)]) for i in range(len(labels.numpy()))])
        return y

    def eval_err(self, label, i):
        targets = self.oneHotEncode(label)
        if self.device.type == 'cuda':
            scores = self.scores[i].data.cpu().numpy()
        else:
            scores = self.scores[i].data.numpy()
        num_samples, num_classes = targets.shape
        err = np.empty((num_samples, num_classes), dtype=np.float32)
        err[:] = np.nan
        pos_labels = np.where(targets >= 0.5)
        neg_labels = np.where(targets < 0.5)
        err[pos_labels] = 1.0 - scores[pos_labels]
        err[neg_labels] = scores[neg_labels]
        self.cur_err_vec[i] = err.astype(np.float32, copy=False)

    def forward(self, x1, x2, x3):
        x1 = torch.reshape(x1, (x1.shape[0], 1, x1.shape[1], x1.shape[2]))
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.convs1]
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]
        x1 = torch.cat(x1, 1)
        norm = x1.norm(p=2, dim=1, keepdim=True)
        x1 = x1.div(norm)

        x2 = torch.reshape(x2, (x2.shape[0], 1, x2.shape[1], x2.shape[2]))
        x2 = [F.relu(conv(x2)).squeeze(3) for conv in self.convs2]
        x2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x2]
        x2 = torch.cat(x2, 1)
        norm = x2.norm(p=2, dim=1, keepdim=True)
        x2 = x2.div(norm)

        x3 = torch.reshape(x3, (x3.shape[0], 1, x3.shape[1], x3.shape[2]))
        x3 = [F.relu(conv(x3)).squeeze(3) for conv in self.convs3]
        x3 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x3]
        x3 = torch.cat(x3, 1)
        norm = x3.norm(p=2, dim=1, keepdim=True)
        x3 = x3.div(norm)

        x = [x1, x2, x3]
        x = torch.cat(x, 1)
        x = self.dropout(x)

        x = [op(x) for op in self.fc1]
        # need to reorder the x1_1 based on self.group and self.indices_list
        if len(x) == 1:
            x = torch.cat(x, 1)
        else:
            indices = np.array(functools.reduce(lambda a, b: a + b, self.fc1.indices_list))
            # get the original order
            original_order = np.argsort(indices)
            if len(original_order) != 1:
                x = [x[index] for index in original_order]

        output = self.layer_forward(x, self.fc2)
        if not self.fc3.group:
            output = torch.cat(output, 1)
        output = self.layer_forward(output, self.fc3)
        output = torch.cat(output, 1)
        return output

    def layer_forward(self, input, layer):
        output = []
        if type(input) == list:
            for i in range(len(input)):
                for j in range(len(layer)):
                    op = layer._modules['{}'.format(j)]
                    if j in layer.indices_list[i]:
                        output.append(op(input[i]))
        else:
            output = [op(input) for op in layer]
        indices = np.array(functools.reduce(lambda x, y: x + y, layer.indices_list))
        # get the original order
        original_order = np.argsort(indices)
        if len(original_order) != 1:
            output = [output[index] for index in original_order]
        return output



    def _update_error_data(self, labels, j):
        """ incrementally add the current training mini-batch to the values """
        index = str(j)
        # decay factor
        error_decay_factor = self.error_decay_factor
        self.eval_err(labels, index)
        cur_err_vec = self.cur_err_vec[index]

        # update mean training error
        cur_err_mean = np.nanmean(cur_err_vec >= 0.5, axis=0)
        # update mean margin
        cur_margin_mean = np.nanmean(cur_err_vec, axis=0)
        # update difficulty probabilities
        cur_diff_vec = (cur_err_vec >= self.margin_mean[index]).astype(np.float64)
        cur_joint_diff_prob = np.dot(np.transpose(cur_diff_vec), cur_diff_vec)/cur_diff_vec.shape[0]
        cur_joint_easy_prob = np.dot(np.transpose(1.0-cur_diff_vec), 1.0-cur_diff_vec)/cur_diff_vec.shape[0]

        sum_err_mean = self.err_div_factor * error_decay_factor * self.err_mean[index] + cur_err_mean
        sum_margin_mean = self.err_div_factor * error_decay_factor * self.margin_mean[index] + cur_margin_mean
        sum_joint_diff_prob = self.err_div_factor * error_decay_factor * self.joint_diff_prob[index] + cur_joint_diff_prob
        sum_joint_easy_prob = self.err_div_factor * error_decay_factor * self.joint_easy_prob[index] + cur_joint_easy_prob
        # update estimates
        self.err_div_factor *= error_decay_factor
        self.err_div_factor += 1
        self.err_mean[index] = sum_err_mean / self.err_div_factor
        self.margin_mean[index] = sum_margin_mean / self.err_div_factor
        self.joint_diff_prob[index] = sum_joint_diff_prob / self.err_div_factor
        self.joint_easy_prob[index] = sum_joint_easy_prob / self.err_div_factor
        # probablity of jointly difficult + probably of joint easy
        self.err_corr[index] = self.joint_easy_prob[index] + self.joint_diff_prob[index]

    def multi_view_clustering(self, layer, max_col):
        # view 1
        task_list_e = self.task_at(layer)
        num_view = self.num_view
        num_branches = len(task_list_e)
        A = np.zeros((num_view, num_branches, num_branches))
        for i in range(num_view):
            # compute error correlation matrix for the group of tasks.
            for j in range(num_branches):
                t_j = task_list_e[j]
                for k in range(num_branches):
                    if k != j:
                        t_k = task_list_e[k]
                        # find the average of correlation between tasks from the two groups
                        A[i, j, k] = np.mean(np.min(self.err_corr[str(i)][t_j, :][:, t_k], axis=1))
                        A[i, k, j] = np.mean(np.min(self.err_corr[str(i)][t_k, :][:, t_j], axis=1))
                    elif k == j:
                        A[i, j, j] = np.mean(np.min(self.err_corr[str(i)][t_j, :][:, t_j], axis=1))
            A[i] = (A[i] + np.transpose(A[i])) / 2.0
        V = dict()
        for i in range(len(A)):
            D = np.diag(A[i].sum(axis=1))
            inv_D = np.linalg.inv(D)
            V['L{}'.format(i)] = np.sqrt(inv_D).dot(A[i]).dot(np.sqrt(inv_D))
        labels = [None for _ in range(1, min(max_col, num_branches) + 1)]
        loss = np.zeros((min(max_col, num_branches),))
        # alpha = max(1.0 - 0.05 * self.cur_round, 0.8)
        aaa = F.relu(self.fc1._modules['0'].weight)
        if self.device.type == 'cuda':
            lambda1 = torch.mean(torch.abs(aaa.data[:, :160])).data.cpu().item()
            lambda2 = torch.mean(torch.abs(aaa.data[:, 160:320])).data.cpu().item()
            lambda3 = torch.mean(torch.abs(aaa.data[:, 320:480])).data.cpu().item()
        else:
            lambda1 = torch.mean(torch.abs(aaa.data[:, :160])).item()
            lambda2 = torch.mean(torch.abs(aaa.data[:, 160:320])).item()
            lambda3 = torch.mean(torch.abs(aaa.data[:, 320:480])).item()
        alpha = self.softmax([lambda1, lambda2, lambda3])
        cof = max(2 - 0.05 * self.cur_round, 1.6)
        for C in range(1, min(max_col, num_branches) + 1):
            # C == 1 is an exception, because we already know the label
            if C == 1:
                labels[0] = np.zeros((num_branches,))
            elif C == num_branches:
                labels[C - 1] = np.arange(num_branches)
            else:
                V1 = self.MV_clustering_representation(V, A[0], C)
                kmeans = KMeans(n_clusters=C, random_state=0)
                sep_cost = np.inf
                for _ in range(cfg.TRAIN.CLUSTER_REP):
                    labels_this = kmeans.fit_predict(V1)
                    sep_cost_this = 0.0
                    for j in range(C):
                        t_j = np.where(labels_this == j)[0]
                        for i in range(len(A)):
                            sep_cost_this += cof * alpha[i] * (1.0 - np.mean(np.min(A[i][t_j, :][:, t_j], axis=1)))/(len(A) * C)
                    if sep_cost_this <= sep_cost:
                        sep_cost = sep_cost_this
                        labels[C - 1] = labels_this
            # add layer-wise cost
            loss[C - 1] += (C - 1) * self.col_cost_at[layer]
            if C != 1 and C != num_branches:
                loss[C - 1] += sep_cost
            else:
                for j in range(C):
                    t_j = np.where(labels[C - 1] == j)[0]
                    for i in range(len(A)):
                        loss[C - 1] += cof * alpha[i] * (1.0 - np.mean(np.min(A[i][t_j, :][:, t_j], axis=1))) / (len(A) * C)
            # display cost information
            print('C={}: Total loss={}, the complexity cost is {}.'.format(C, loss[C - 1], (C - 1) * self.col_cost_at[layer]))

        # find out the minimum cost partition
        loss = loss[:len(loss) - 1]
        Cmin = np.argmin(loss) + 1
        partition = [list(np.where(labels[Cmin - 1] == c)[0]) for c in range(Cmin)]
        return partition

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    def MV_clustering_representation(self, V, A1, k):
        N = A1.shape[0]
        # U1 is eigenvector, E1 is the corresponding eigenvalue
        E = []
        U = dict()
        for i in range(len(V)):
            E1, U1 = LA.eigh(V['L{}'.format(i)])
            E.append(E1)
            U['U{}'.format(i)] = U1
        lambda_1 = 0.1
        numiter = 45
        i = 2
        while (i <= numiter + 1):
            for j in range(1, len(V)):
                L_component = np.zeros((N, N))
                for k in range(len(V)):
                    if k == j:
                        continue
                    L_component = L_component + U['U%s' % k].dot(U['U%s' % k].transpose())
                L_component = (L_component + L_component.transpose()) / 2
                E2, U['U%s' % j] = LA.eigh(U['U%s' % j] + lambda_1 * L_component)
            L_component = np.zeros((N, N))
            for j in range(len(V)):
                if k == j:
                    continue
                L_component = L_component + U['U%s' % k].dot(U['U%s' % k].transpose())
            L_component = (L_component + L_component.transpose()) / 2
            E2, U['U%s' % j] = LA.eigh(U['U%s' % j] + lambda_1 * L_component)
            i = i + 1
        U_norm =dict()
        for a in range(len(U)):
            normvect = np.sqrt(np.diag(U['U{}'.format(a)].dot(U['U{}'.format(a)].transpose())))
            for x in np.argwhere(normvect == 0.0):
                for i, j in x:
                    normvect[i, j] = 1
            U_norm['{}'.format(a)] = np.linalg.inv(np.diag(normvect)).dot(U['U{}'.format(a)])
        sum = U_norm['0']
        for i in range(1, len(U_norm)):
            sum = sum + U_norm['{}'.format(i)]
        V1 = sum / len(U_norm)
        normvect = np.sqrt(np.diag(V1.dot(V1.transpose())))
        normvect[np.where(normvect == 0.0)] = 1
        V1 = np.linalg.inv(np.diag(normvect)).dot(V1)
        return V1

    def task_at(self, i):
        return self.net_graph[i]

    def improve_model(self):
        """ Make the current model larger by creating columns at lower layers
            Return "model_improved"
        """
        active_layer = self.cur_layer - 1
        # cannot create a branch when the active layer is already 0.
        if active_layer == 0:
            return False
        partition = self.multi_view_clustering(active_layer, self.num_class)
        if len(partition) > 1:
            self.net_graph[active_layer-1] = list(partition)
            name = self.layer_list[active_layer-1]
            if active_layer + 1 == self.num_layers:
                cur_layer = getattr(self, self.layer_list[active_layer])
                cur_layer.group = True
                cur_layer.indices_list = partition
            else:
                cur_layer = getattr(self, self.layer_list[active_layer])
                cur_layer.group = True
                cur_layer.indices_list = partition
            self.update_structure(partition, name)
            self.cur_layer = self.cur_layer - 1
            return True
        else:
            return False


    def update_structure(self, partition, name):
        # num_class = self.num_class
        cur_layer = self.__getattr__(name)
        for i in range(len(partition)-1):
            output_features, input_features = cur_layer._modules['0'].weight.shape
            new_model = Linear(input_features, output_features)
            new_model = new_model.double().to(self.device)
            cur_layer.add_module('{}'.format(i+1), new_model)


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    def backward(self, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


    def save_for_backward(self, *tensors):
        r"""Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors
