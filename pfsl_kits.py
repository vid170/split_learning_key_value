import os
import random
import string
import socket
import requests
import sys
import threading
import time
import torch
from models.nnUNet.nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from math import ceil
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils.split_dataset import split_dataset, split_dataset_cifar10tl_exp
from utils.client_simulation_kits import generate_random_clients
from utils.connections import send_object
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import multiprocessing
from sklearn.metrics import classification_report
import torch.optim as optim 
import copy
from datetime import datetime
from scipy.interpolate import make_interp_spline
import numpy as np
from ConnectedClient_kits import ConnectedClient
import importlib
from utils.merge import merge_grads, merge_weights
import pandas as pd
import time 
from utils import dataset_settings, datasets
import torch.nn.functional as F
import os 
import sys
sys.path.append(os.path.join(os.getcwd(), ".."))
from FLamby.flamby.datasets.fed_kits19 import FedKits19


from models.nnUNet.nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin, Generic_UNet
from models.nnUNet.nnunet.network_architecture.initialization import InitWeights_He
from torch import nn
from models import nnunet_final


class Baseline(Generic_UNet):
    def __init__(self):
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
        conv_kernel_sizes = [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ]
        super(Baseline, self).__init__(
            input_channels=1,
            base_num_features=32,
            num_classes=3,
            num_pool=5,
            num_conv_per_stage=2,
            feat_map_mul_on_downscale=2,
            conv_op=nn.Conv3d,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=nn.Dropout3d,
            dropout_op_kwargs={"p": 0, "inplace": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=False,
            dropout_in_localization=False,
            final_nonlin=lambda x: x,
            weightInitializer=InitWeights_He(1e-2),
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            conv_kernel_sizes=conv_kernel_sizes,
            upscale_logits=False,
            convolutional_pooling=True,
            convolutional_upsampling=True,
            max_num_features=None,
            basic_block=ConvDropoutNormNonlin,
        )


class BaselineLoss(DC_and_CE_loss):
    def __init__(
        self,
        soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False},
        ce_kwargs={},
        aggregate="sum",
        square_dice=False,
        weight_ce=1,
        weight_dice=1,
        log_dice=False,
        ignore_label=None,
    ):
        super(BaselineLoss, self).__init__(
            soft_dice_kwargs=soft_dice_kwargs,
            ce_kwargs=ce_kwargs,
            aggregate=aggregate,
            square_dice=square_dice,
            weight_ce=weight_ce,
            weight_dice=weight_dice,
            log_dice=log_dice,
            ignore_label=ignore_label,
        )



def calculate_train_acc_kits(outputs, targets):
    y_pred = outputs.detach().cpu()
    # print(y_pred.shape, "---", type(y_pred))
    preds_softmax = F.softmax(y_pred, 1)
    # F.softmax(x, 1)
    preds = preds_softmax.argmax(1)
    y =     targets.detach().cpu()
    gt=y
    gt = gt.float()
    preds = preds.float()
    # print(type(gt), "....", type(preds))
    
    tk_pd2 = torch.gt(preds, 0).float()
    tk_gt = torch.gt(gt, 0).float()
   
    num = 2 * (tk_pd2 * tk_gt).sum()
    eps=1e-5
    denom = tk_pd2.sum() + tk_gt.sum() + eps
    tk_dice=num/denom
    
    tk_pd2=(preds==2).float()
    tk_gt=(gt==2).float()
    num = 2 * (tk_pd2 * tk_gt).sum()
    eps=1e-5
    denom = tk_pd2.sum() + tk_gt.sum() + eps
    tu_dice=num/denom
    dice_score=(tk_dice+tu_dice)/2
    return dice_score


# from FLamby.flamby.datasets.fedkits.dataset import FedKits19


#To load train and test data for each client for setting 1 and setting 2
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        print("idxs: ", idxs)
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


#To intialize every client with their train and test data for setting 4
def initialize_client(client, dataset, batch_size, test_batch_size, tranform):
    
    client.load_data(dataset, transform)
    print(f'Length of train dataset client {client.id}: {len(client.train_dataset)}')
    client.create_DataLoader(batch_size, test_batch_size)


#Plots class distribution of train data available to each client
def plot_class_distribution(clients, dataset, batch_size, epochs, opt, client_ids):
    class_distribution=dict()
    number_of_clients=len(client_ids)
    if(len(clients)<=20):
        plot_for_clients=client_ids
    else:
        plot_for_clients=random.sample(client_ids, 20)
    
    fig, ax = plt.subplots(nrows=(int(ceil(len(client_ids)/5))), ncols=5, figsize=(15, 10))
    j=0
    i=0

    #plot histogram
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        class_distribution[client_id]=df['labels'].value_counts().sort_index()
        df['labels'].value_counts().sort_index().plot(ax = ax[i,j], kind = 'bar', ylabel = 'frequency', xlabel=client_id)
        j+=1
        if(j==5 or j==10 or j==15):
            i+=1
            j=0
    fig.tight_layout()
    plt.show()
    
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  
    plt.savefig('plot_setting3_exp.png')

    max_len=0
    #plot line graphs
    for client_id in plot_for_clients:
        df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
        max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    plt.xticks(np.arange(0,10))
    plt.ylim(0, max_len)
    plt.legend()
    plt.show()
    
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution



if __name__ == "__main__":    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)
                             
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    # train_dataset_size, input_channels = split_dataset(args.dataset, client_ids, pretrained=args.pretrained)
    

    print(f'Random client ids:{str(client_ids)}')
    transform=None
    max_epoch=0
    max_f1=0
    max_accuracy=0

    #Assigning train and test data to each client depending for each client
    print('Initializing clients...')
    client_idx=0
    print("clients: ", clients)
    for _, client in clients.items():
        client.train_dataset=FedKits19(center=client_idx,train=True , pooled=False)
        client.test_dataset=FedKits19(center=client_idx, train=False, pooled=False)
        client.create_DataLoader(args.batch_size, args.test_batch_size
        )
        x,y=next(iter(client.train_DataLoader))
       
        client_idx+=1
        print("client_idx: ", client_idx)
    
    print('Client Intialization complete.')    
    # Train and test data intialisation complete

    #Setting the start of personalisation phase
    if(args.setting!='setting2'):
        args.checkpoint=args.epochs+10

    input_channels=1
    # class_distribution=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, client_ids)


    #Assigning front, center and back models and their optimizers for all the clients
    model = importlib.import_module(f'models.{args.model}')
  
    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        # client.back_model
        
    print('Done')
  
    for _, client in clients.items():
        # client.front_optimizer = optim.SGD(client.front_model.parameters(), lr=args.lr, momentum=0.9)
        # client.back_optimizer = optim.SGD(client.back_model.parameters(), lr=args.lr, momentum=0.9)
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)

    first_client = clients[client_ids[0]]
    client_idx=0
    
    num_iterations = ceil(len(first_client.train_DataLoader.dataset)/args.batch_size)

    num_test_iterations= ceil(len(first_client.test_DataLoader.dataset)/args.test_batch_size)
    sc_clients = {} #server copy clients

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    # for _,s_client in sc_clients.items():
    #     s_client.center_model = model.center(pretrained=args.pretrained)
    #     s_client.center_model.to(device)
    #     # s_client.center_optimizer = optim.SGD(s_client.center_model.parameters(), lr=args.lr, momentum=0.9)
    #     s_client.center_optimizer = optim.Adam(s_client.center_model.parameters(), args.lr)

    st = time.time()

    macro_avg_f1_2classes=[]

    criterion=F.cross_entropy
    max_acc=0

    # for epoch in range(args.epochs):
    #     print("Epoch: ", epoch)
    #     model=nnunet_final.get_model()
    #     model_opt=optim.Adam(model.parameters(), lr=args.lr)
    #     iterator=iter(clients[client_ids[0]].train_DataLoader)
    #     test_iterator=iter(clients[client_ids[0]].test_DataLoader)
    #     model.to(device)
    #     acc=0
    #     for it in range(num_iterations):
    #         data, targets = next(iterator)
    #         data, targets = data.to(device), targets.to(device)
    #         op=model(data)
    #         bloss=BaselineLoss()
    #         loss=bloss(op, targets)
    #         loss.backward()
    #         model_opt.zero_grad()
    #         acc+=calculate_train_acc_kits(op, data)
    #     acc=acc/num_iterations
    #     print("train acc for this epoch: ", acc)
    #     acc=0
    #     for it in range(num_test_iterations):
    #         data, targets = next(test_iterator)
    #         data, targets = data.to(device), targets.to(device)
    #         op=model(data)
    #         acc+=calculate_train_acc_kits(op, data)
    #     acc=acc/num_test_iterations
    #     print("test acc for this epoch: ", acc)
    #     if(acc> max_acc):
    #         max_acc=acc
    #         print("max acc is : ", max_acc)
    #     torch.save({'model_wts': model.state_dict()}, 'saved_models/model_checkpoint1.pkl')
    
    # #Starting the training process 
    for epoch in range(args.epochs):
        if(epoch==args.checkpoint): # When starting epoch of the perosnalisation is reached, freeze all the layers of the center model 
            print("freezing the center model")
            for _, s_client in sc_clients.items():
                s_client.center_model.freeze(epoch, pretrained=True)

        overall_train_acc.append(0)


        for _, client in clients.items():
            client.train_acc.append(0)
            client.iterator = iter(client.train_DataLoader)
            
        #For every batch in the current epoch
        for iteration in range(num_iterations):
            print(f'\rEpoch: {epoch+1}, Iteration: {iteration+1}/{num_iterations}', end='')
            for _, client in sc_clients.items():
                client.skips=[]

            for _, client in clients.items():
                client.forward_front()
           
            for client_id, client in sc_clients.items():
                client.remote_activations1 = clients[client_id].remote_activations1
                client.skips=[]
                client.skips.append(clients[client_id].remote_activations1)
                client.center_model = model.center(pretrained=args.pretrained, skips=client.skips)
                client.center_model.to(device)
                client.center_optimizer = optim.Adam(client.center_model.parameters(), args.lr)

                client.forward_center()
                
            client_idx=0

            for client_id, client in clients.items():
                client.back_model = model.back(pretrained=args.pretrained, skips=sc_clients[client_id].skips)
                client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)
                client.remote_activations2 = sc_clients[client_id].remote_activations2
                client.forward_back()
                client_idx+=1
                
            for _, client in clients.items():
                client.calculate_loss()

            for _, client in clients.items():
                client.loss.backward()
                
            # for _, client in clients.items():
                # client.backward_back()
                
            for client_id, client in sc_clients.items():
                client.activations2 = clients[client_id].remote_activations2
                client.backward_center()

            for _, client in clients.items():
                client.step_back()
                client.zero_grad_back()
                

            #merge grads uncomment below

            # if epoch%2 == 0:
            #     params = []
            #     normalized_data_sizes = []
            #     for iden, client in clients.items():
            #         params.append(sc_clients[iden].center_model.parameters())
            #         normalized_data_sizes.append(len(client.train_dataset) / train_dataset_size)
            #     merge_grads(normalized_data_sizes, params)

            for _, client in sc_clients.items():
                client.center_optimizer.step()
                client.center_optimizer.zero_grad()
                # break

            for _, client in clients.items():
                acc=client.calculate_train_acc_kits()
                client.train_acc[-1] += acc #train accuracy of every client in the current epoch in the current batch
                print("train acc per iteration: ", acc)
                

        for c_id, client in clients.items():
            # client.train_acc[-1] /= num_iterations # train accuracy of every client of all the batches in the current epoch
            client.train_acc[-1]/=num_iterations
            overall_train_acc[-1] += client.train_acc[-1] 

        overall_train_acc[-1] /= len(clients) #avg train accuracy of all the clients in the current epoch
        print("acc: ", overall_train_acc[-1])
        

        # merge weights below uncomment 
        params = []
        for _, client in sc_clients.items():
            params.append(copy.deepcopy(client.center_model.state_dict()))
        w_glob = merge_weights(params)

        for _, client in sc_clients.items():
            client.center_model.load_state_dict(w_glob)

        params = []

        #In the personalisation phase merging of weights of the back layers is stopped
        if(epoch <=args.checkpoint):
            for _, client in clients.items():
                params.append(copy.deepcopy(client.back_model.state_dict()))
            w_glob_cb = merge_weights(params)
            del params
    
            for _, client in clients.items():
                client.back_model.load_state_dict(w_glob_cb)

        #Testing every epoch
        if (epoch%1 == 0 ):
            if(epoch==args.checkpoint):
                print("freezing the center model")
                
                
                for _, s_client in sc_clients.items():
                    s_client.center_model.freeze(epoch, pretrained=True)
            with torch.no_grad():
                print("testing")
                test_acc = 0
                overall_test_acc.append(0)
            
                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.iterator = iter(client.test_DataLoader)
                    client.pred=[]
                    client.y=[]

                #For every batch in the testing phase
                for iteration in range(num_test_iterations):

                    for _, client in sc_clients.items():
                        client.skips=[]

                    for _, client in clients.items():
                        client.forward_front()
           
                    for client_id, client in sc_clients.items():
                        client.remote_activations1 = clients[client_id].remote_activations1
                        client.skips=[]
                        # print("len skips is : ", len(client.skips))
                        client.skips.append(clients[client_id].activations1)
                        client.center_model = model.center(pretrained=args.pretrained, skips=client.skips)
                        client.center_model.to(device)
                        client.center_optimizer = optim.Adam(client.center_model.parameters(), args.lr)
                        client.forward_center()

                    for client_id, client in clients.items():
                        client.back_model = model.back(pretrained=args.pretrained, skips=sc_clients[client_id].skips)
                        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)
                        client.remote_activations2 = sc_clients[client_id].remote_activations2
                        client.forward_back()
                        # client_idx+=1
                
    
                    # for client_id, client in sc_clients.items():
                    #     client.remote_activations1 = clients[client_id].remote_activations1
                    #     client.forward_center()

                    # for client_id, client in clients.items():
                    #     client.remote_activations2 = sc_clients[client_id].remote_activations2
                    #     client.forward_back()

                    for _, client in clients.items():
                        test_acc=client.calculate_test_acc_kits()
                        print("testing accuracy per iteration: ", test_acc)
                        client.test_acc[-1] += test_acc

                for _, client in clients.items():
                    client.test_acc[-1] /= num_test_iterations
                    overall_test_acc[-1] += client.test_acc[-1]
                    #Calculating the F1 scores using the classification report from sklearn metrics
                    if(args.setting=='setting2'):
                        clr=classification_report(np.array(client.y), np.array(client.pred), output_dict=True, zero_division=0)
                        idx=client_idxs[_]
                        macro_avg_f1_2classes.append((clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2) #macro f1 score of the 2 prominent classes in setting2
                        
                overall_test_acc[-1] /= len(clients) #average test accuracy of all the clients in the current epoch

                if(args.setting=='setting2'):
                    f1_avg_all_user=sum(macro_avg_f1_2classes)/len(macro_avg_f1_2classes) #average f1 scores of the clients for the prominent 2 classes in the current epoch
                    macro_avg_f1_2classes=[]
                    
                    #Noting the maximum f1 score
                    if(f1_avg_all_user> max_f1):
                        max_f1=f1_avg_all_user
                        max_epoch=epoch
                        
                else:
                    print("test accuracy: ", overall_test_acc[-1])
                    if(overall_test_acc[-1]> max_accuracy):
                        max_accuracy=overall_test_acc[-1]
                        max_epoch=epoch
                        print("MAX test accuracy: ", max_accuracy)
                        
    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    '''

    et = time.time()
    print("\nTraining Accuracy: ", overall_train_acc[max_epoch])
    if(args.setting=='setting2'):
        print("Maximum F1 Score: ", max_f1)
    else:
        print("Maximum Test Accuracy: ", max_accuracy)
    print(f"Time taken for this run {(et - st)/60} mins")
    
    # calculating the train and test standarad deviation and teh confidence intervals 
    X = range(args.epochs)
    all_clients_stacked_train = np.array([client.train_acc for _,client in clients.items()])
    all_clients_stacked_test = np.array([client.test_acc for _,client in clients.items()])
    epochs_train_std = np.std(all_clients_stacked_train,axis = 0, dtype = np.float64)
    epochs_test_std = np.std(all_clients_stacked_test,axis = 0, dtype = np.float64)

    #Y_train is the average client train accuracies at each epoch
    #epoch_train_std is the standard deviation of clients train accuracies at each epoch
    Y_train = overall_train_acc
    Y_train_lower = Y_train - (1.65 * epochs_train_std) #95% of the values lie between 1.65*std
    Y_train_upper = Y_train + (1.65 * epochs_train_std)

    Y_test = overall_test_acc
    Y_test_lower = Y_test - (1.65 * epochs_test_std) #95% of the values lie between 1.65*std
    Y_test_upper = Y_test + (1.65 * epochs_test_std)

    Y_train_cv =  epochs_train_std / Y_train
    Y_test_cv = epochs_test_std / Y_test

    plt.figure(0)
    plt.plot(X, Y_train)
    plt.fill_between(X,Y_train_lower , Y_train_upper, color='blue', alpha=0.25)
    # plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    plt.show()
    

    plt.figure(1)
    plt.plot(X, Y_test)
    plt.fill_between(X,Y_test_lower , Y_test_upper, color='blue', alpha=0.25)
    # plt.savefig(f'./results/test_acc_vs_epoch/{args.dataset}_{args.number_of_clients}clients_{args.epochs}epochs_{args.batch_size}batch_{args.opt}.png', bbox_inches='tight')
    plt.show()
    

    plt.figure(2)
    plt.plot(X, Y_train_cv)
    plt.show()
 

    plt.figure(3)
    plt.plot(X, Y_test_cv)
    plt.show()
