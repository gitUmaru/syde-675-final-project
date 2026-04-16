import random
import numpy as np
import os
import pickle
import time
import scipy.io as sio
import torch
import torch.nn as nn
import utils
from model import AutoEncoder
import pandas as pd

from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
"""
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# 这里是为了确保每次运行此程序后结果一致，别的代码也不用加
"""

class Train_test:
    def __init__(self, dataset, device, dr_type="original"):
        super(Train_test, self).__init__()
        self.dr_type = dr_type
        self.device = device
        self.dataset = dataset
        self.data = utils.Data(dataset, device)
        self.P, self.L, self.col = self.data.get("num_endmembers"), self.data.get("num_bands"), self.data.get(
            "num_cols")
        self.loader = self.data.get_loader(batch_size=self.col ** 2)
        self.is_mars = False
        if dataset == 'Samson' or dataset == 'Jasper':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.02
            self.weight_decay_param = 0
        elif dataset == 'dc_test':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 1e-5
        elif dataset == 'Urban4_new':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 1e-4
        elif dataset == 'sim1020':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 1000, 0.03
            self.weight_decay_param = 3e-5
        elif dataset == 'berlin_test':
            self.LR, self.EPOCH = 3e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 0
        elif dataset == 'apex':
            self.LR, self.EPOCH = 9e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 4000, 0.05
            self.weight_decay_param = 4e-5
        elif dataset == 'moni30':
            self.LR, self.EPOCH = 4e-4, 200
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 3000, 0.05
            self.weight_decay_param = 4e-5
        elif dataset == 'houston_lidar':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 4e-5
        elif dataset == 'moffett':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 0, 0.03
            self.weight_decay_param = 4e-5
        elif dataset == 'muffle':
            self.LR, self.EPOCH = 5e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 2000, 0.03
            self.weight_decay_param = 4e-5
        elif dataset == 'holden':
            self.LR, self.EPOCH = 9e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 4000, 0.05
            self.weight_decay_param = 4e-5
            self.is_mars = True
        elif dataset == 'nili':
            self.LR, self.EPOCH = 9e-3, 240
            self.patch, self.dim = 5, 200
            self.beta, self.gamma = 4000, 0.05
            self.weight_decay_param = 4e-5
            self.is_mars = True
        
        self.L_reduced = self.L
        self.reduced_idcs = list[range(self.L_reduced)]
    
    def kmeans_reduce(self, data):
        data = data.T
        n_bands, n_pixels = data.shape
        k = self.L_reduced # Number of bands to select

        # 1. Cluster the bands
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # 2. Select the best band from each cluster
        selected_bands = []
        for i in range(k):
            # Find bands in the current cluster
            band_indices = np.argwhere(labels == i).flatten()
            
            # Calculate distance to centroid for all bands in cluster
            diffs = data[band_indices] - centers[i]
            distances = np.linalg.norm(diffs, axis=1)
            
            # Select index of band closest to centroid
            optimal_band_idx = int(band_indices[np.argmin(distances)])
            selected_bands.append(optimal_band_idx)

        # Sort selected bands
        selected_bands.sort()
        self.reduced_idcs = selected_bands
        # reduced_data = data[selected_bands, :]
        # print(selected_bands)
        # return reduced_data

    def reduce_dim(self, x):
        # rng = np.random.default_rng(seed=0)
        # idcs = rng.choice(self.L, size=self.L_reduced, replace=False)
        # idcs = list(range(self.L_reduced))
        idcs = self.reduced_idcs
        x = x[:, idcs]
        # x = (pca.transform(x))
        return x
    
    def run(self, num_runs):
        end = []
        abu = []
        r = []
        time_start = time.time()

        output_path = '../Results'
        method_name = 'TAEU'
        mat_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + self.dr_type + str(self.L_reduced)
        endmember_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + 'endmember'
        abundance_folder = output_path + '/' + method_name + '/' + self.dataset + '/' + 'abundance'
        if not os.path.exists(mat_folder):
            os.makedirs(mat_folder)
        if not os.path.exists(endmember_folder):
            os.makedirs(endmember_folder)
        if not os.path.exists(abundance_folder):
            os.makedirs(abundance_folder)

        for run in range(1, num_runs + 1):
            """
            seed = 1
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            # 这里是为了确保每次循环run后结果一致
            # 代码在GPU上运行，还需要设置torch.backends.cudnn.deterministic = True，以确保使用相同的输入和参数时，CUDA卷积运算的结果始终是确定的。
            # 其它有的代码如CyCUNet里没加这行（有卷积运算）也能保证随机性一致，不知道为啥
            # 这将会禁用一些针对性能优化的操作，因此可能会导致训练速度变慢
            """

            x = self.data.get("hs_img")
            # pca = PCA(n_components=n_reduced_bands)
            # pca.fit(x).to(self.device)
            self.kmeans_reduce(x.cpu())

            net = AutoEncoder(P=self.P, L=self.L_reduced, size=self.col,
                              patch=self.patch, dim=self.dim).to(self.device)

            total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f"Total number of parameters: {total_params}")

            net.apply(net.weights_init)
            # 首先，net.apply(net.weights_init) 调用了 net 的 weights_init 方法来对神经网络 net 的权重进行初始化。
            # 然后，代码通过 net.state_dict() 方法获取了神经网络 net 的所有状态字典，并将其存储在 model_dict 变量中。
            # 接下来，代码修改了 model_dict 中的 decoder.0.weight 参数的值，将其设置为 self.init_weight。
            # 这里的 decoder.0.weight 是指神经网络 net 中第一层解码器的权重。
            # 最后，代码通过 net.load_state_dict(model_dict) 方法将修改后的状态字典重新加载到神经网络 net 中，使得神经网络的权重被初始化为新的值。
            model_dict = net.state_dict()

            self.init_weight = self.data.get("init_weight")
            self.init_weight = self.reduce_dim(self.init_weight.T).T
            self.init_weight = self.init_weight.unsqueeze(2).unsqueeze(3).float()

            model_dict['decoder.0.weight'] = self.init_weight
            net.load_state_dict(model_dict)

            loss_func = nn.MSELoss(reduction='mean')
            loss_func2 = utils.SAD(self.L_reduced)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
            apply_clamp_inst1 = utils.NonZeroClipper()

            endmember_name = self.dataset + '_run' + str(run)
            endmember_path = endmember_folder + '/' + endmember_name
            abundance_name = self.dataset + '_run' + str(run)
            abundance_path = abundance_folder + '/' + abundance_name
            abundanceAndGT_name = self.dataset + 'withGT_run' + str(run)

            net.train().cuda()
            print('Start training!', 'run:', run)
            for epoch in range(self.EPOCH):
                for i, x in enumerate(self.loader):
                    x = self.reduce_dim(x)

                    x = x.transpose(1, 0).view(1, -1, self.col, self.col).cuda()
                    abu_est, re_result = net(x)

                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L_reduced, -1).transpose(1, 2),
                                          x.view(1, self.L_reduced, -1).transpose(1, 2))
                    loss_sad = self.gamma * torch.sum(loss_sad).float()
                    ab = abu_est.view(-1, self.col * self.col)
                    # osp = utils.OSP(ab, self.P)
                    # loss3 = torch.sum(torch.pow(torch.abs(ab) + 1e-8, 0.8))
                    total_loss = loss_re + loss_sad

                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    total_norm = nn.utils.get_total_norm([p.grad for p in net.parameters() if p.grad is not None], norm_type=1)
                    if not torch.isnan(total_norm):
                        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    else:
                        print("Grad Norm:", total_norm)
                        optimizer.zero_grad()

                    optimizer.step()

                    net.decoder.apply(apply_clamp_inst1)

                    if epoch % 20 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)

                scheduler.step()

            print('-' * 70)

            # Testing ================
            net.eval()

            x = self.data.get("hs_img")
            x = self.reduce_dim(x)
            x = x.transpose(1, 0).view(1, -1, self.col, self.col)

            abu_est, re_result = net(x)  # abuest 1 3 95 95

            abu_est = abu_est.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()

            true_endmem = self.reduce_dim(self.data.get("end_mem").cpu().numpy())

            true_abundance = self.data.get("abd_map").cpu().numpy()
            est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
            est_endmem = est_endmem.reshape((self.L_reduced, self.P))
            est_endmem = est_endmem.T
            utils.plotEndmembersAndGT(est_endmem, true_endmem, endmember_path, end)
            utils.plotAbundancesSimple(abu_est, true_abundance, abundance_path, abu)
            mat_path = mat_folder + '/' + method_name + '_run' + str(run) + '.mat'
            sio.savemat(mat_path, {'e_est': est_endmem, 'a_est': abu_est, 'e_true': true_endmem, 'a_true': true_abundance})

            x = x.view(-1, self.col * self.col).permute(1, 0).detach().cpu().numpy()
            re_result = re_result.view(-1, self.col * self.col).permute(1, 0).detach().cpu().numpy()
            armse_y = np.sqrt(np.mean(np.mean((re_result - x) ** 2, axis=1)))
            r.append(armse_y)

        end = np.reshape(end, (-1, self.data.get("num_endmembers") + 1))
        abu = np.reshape(abu, (-1, self.data.get("num_endmembers") + 1))
        dt = pd.DataFrame(end)
        dt2 = pd.DataFrame(abu)
        dt3 = pd.DataFrame(r)
        dt.to_csv(
            mat_folder + '/' + self.dataset + 'SAD.csv')
        dt2.to_csv(
            mat_folder + '/' + self.dataset + 'RMSE.csv')
        dt3.to_csv(mat_folder + '/' + self.dataset + 'RE.csv')
        # abundanceGT_path = output_path + '/' + method_name + '/' + self.dataset + '/' + self.dataset + 'AGT'
        abundance_compare_path = mat_folder + '/' + self.dataset + 'ACompare'
        endmember_compare_path = mat_folder + '/' + self.dataset + 'ECompare'

        data = sio.loadmat(mat_path)
        est_endmem, abu_est, true_endmem, true_abundance = (data['e_est'], data['a_est'], data['e_true'], data['a_true'])

        if self.is_mars:
            utils.plot_mars(abu_est, true_abundance, abundance_compare_path)
        else:
            utils.plot_abundance_comparison(abu_est, true_abundance, abundance_compare_path)
            utils.plot_endmember_comparison(est_endmem, true_endmem, endmember_compare_path)
        time_end = time.time()
        print('Runtime:', time_end - time_start)


tmod = Train_test(dataset='apex', device=device)  # 要用Urban4 new 波段数必须是超参数patch(5)的倍数
tmod.run(num_runs=1)
