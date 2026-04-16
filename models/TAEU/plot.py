import utils
import scipy.io as sio
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = 'nili'
data = utils.Data(dataset, device)
P, L, col = data.get("num_endmembers"), data.get("num_bands"), data.get(
    "num_cols")

if dataset in ('holden', 'nili'):
    is_mars = True
run = 1
L_reduced = L
dr_type = 'original'

output_path = '../Results'
method_name = 'TAEU'
mat_folder = output_path + '/' + method_name + '/' + dataset + '/' + dr_type + str(L_reduced)
endmember_folder = output_path + '/' + method_name + '/' + dataset + '/' + 'endmember'
abundance_folder = output_path + '/' + method_name + '/' + dataset + '/' + 'abundance'
abundance_compare_path = mat_folder + '/' + dataset + 'ACompare'
endmember_compare_path = mat_folder + '/' + dataset + 'ECompare'

mat_path = mat_folder + '/' + method_name + '_run' + str(run) + '.mat'

data = sio.loadmat(mat_path)
est_endmem, abu_est, true_endmem, true_abundance = (data['e_est'], data['a_est'], data['e_true'], data['a_true'])

if is_mars:
    # classmap = 0#data.classmap
    utils.plot_mars(abu_est, true_abundance, abundance_compare_path)
else:
    utils.plot_abundance_comparison(abu_est, true_abundance, abundance_compare_path)
    utils.plot_endmember_comparison(est_endmem, true_endmem, endmember_compare_path)