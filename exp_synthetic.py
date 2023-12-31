import numpy as np
import torch
from model.hgcrm import HGCRM,  train_model_adam ,train_model_gist
import matplotlib.pyplot as plt
from sklearn import preprocessing # Normalization
from sklearn.metrics import confusion_matrix
import os

saved_data_dir = './generated_datasets/'

def multi_exp(A_p,T,seed,lam_nonsmooth=0.3):
    """
    Run the experiment for given setting
    Args:
        A_p: the size of each group
        T: the length of generated time series
        seed:
        lam_nonsmooth: parameter for nonsmooth regularization when training the model
    """
    print("Now train under setting: A_p={}, T={}, seed={}".format(A_p,T,seed))
    #print enviorment information 
    print("torch.__version__:",torch.__version__)
    if torch.cuda.is_available():
        print("torch.version.cuda:",torch.version.cuda)
        print("torch.backends.cudnn.version():",torch.backends.cudnn.version())
        print("torch.cuda.get_device_name(0):",torch.cuda.get_device_name(0))

    #set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # For GPU acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Simulate data
    saved_file_name = f'A_p_{A_p}-T_{T}-seed_{seed}.npz'
    saved_data_file = os.path.join(saved_data_dir, saved_file_name)
    npz_data = np.load(saved_data_file)
    X_np, GC = npz_data['X'], npz_data['GC']

    A_p = torch.tensor(A_p,dtype=torch.int, device=device) 

    CovX = torch.tensor(np.cov(X_np.T),dtype=torch.double,device=device)
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

    m = len(A_p) # Number of series

    # Set up model
    hgcrm = HGCRM(m, hidden=10).cuda(device=device) if torch.cuda.is_available() else HGCRM(m, hidden=10)
    check_every = 100
    train_loss_list,Y,A = train_model_adam(hgcrm, X, CovX, A_p, lr=1e-5, check_every=check_every)


    # Train Y with GIST using HGCRM
    check_every = 100
    train_loss_list, train_mse_list = train_model_gist(hgcrm, Y, lam=lam_nonsmooth, lam_ridge=1e-4, lr=0.005,
                                                       max_iter=5000, check_every=check_every)  #, truncation=5)

    # Verify learned Granger causality
    GC_est = hgcrm.GC().cpu().data.numpy()

    print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
    print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))

    confusion = confusion_matrix(GC.reshape(1,-1)[0], GC_est.reshape(1,-1)[0])

    # save the result
    tp = confusion[1,1]
    fp = confusion[0,1]
    fn = confusion[1,0]
    f1 = 2*tp/(2*tp+fp+fn)

    context = np.vstack((GC,GC_est))
    np.savetxt("results/Comparision_{}_{}_{}_{}%.txt".format(A_p.cpu().data.numpy(), T, seed,100*f1), context)



if __name__ == "__main__":


    multi_exp(A_p = [3,4,5,3,4,5,3,5],  T = 1500, seed=s,lam_nonsmooth=0.4)
