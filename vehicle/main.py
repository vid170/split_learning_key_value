import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryHingeLoss
from hinge_loss import HingeLoss
from model import LinearModel
from data import Vehicle, load_vehicle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import argparse
import wandb
import time
import copy
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import classification_report



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
def parse_arguments():
    parser=argparse.ArgumentParser(
        description="Cross Silo FL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed"
    )
    parser.add_argument(
        "--disable_wandb",
        type=bool,
        default=False,
        help='Disable wandb'
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        metavar="sv",
        help="Whether the global model is to be saved or not"
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Total number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        metavar="LR",
        help="Learning rate",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="mnist",
    #     help="States dataset to be used",
    # )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1024,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=512,
        help="Input batch size for testing"

    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=64,
        help="Input batch size for testing"

    )
    parser.add_argument(
        "--local_ep",
        type=int,
        default=1,
        help="Number of local steps before averaging"
    )
    args = parser.parse_args()
    args = parser.parse_args()
    return args


def calculate_accuracy(preds, y):
    # preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 *correct.float()/preds.shape[0]
    return acc

def plot_class_distribution(clients,  client_ids):
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
    # wandb.log({"Histogram": wandb.Image(plt)})
    plt.savefig('plot_fl.png')
    # plt.savefig(f'./results/classvsfreq/settin3{dataset}.png')  

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
    # wandb.log({"Line graph": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_fre/q/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return class_distribution

def val_test_split(dataset_obj,seed,test_size=0.15,val_size=0.15):
    x=dataset_obj.X
    y=dataset_obj.Y
    train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=test_size,random_state=seed)

    train_x,val_x, train_y, val_y=train_test_split(train_x,train_y, test_size=val_size/(1-test_size),random_state=seed)

    train=Vehicle(train_x, train_y)
    test=Vehicle(test_x,test_y)
    val=Vehicle(val_x,val_y)
    return train,val,test

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgM(w,momentum=0.9,lr=0.01):
    w_avg=copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[k]+=w[i][k]
        w_avg[k]=torch.div(w_avg[k],len(w))
    v=copy.deepcopy(w_avg)
    for k in v.keys():
        v[k]=w_avg[k]+v[k]*momentum
    w_new=copy.deepcopy(w_avg)
    for k in w_new.keys():
        w_new[k]=w_avg[k]-v[k]*lr
    return w_new
    


class LocalUpdate(object):
    def __init__(self,idx,lr,device, local_ep ,batch_size,val_batch_size,dataset_train=None, dataset_val=None):
        self.idx=idx
        self.device=device
        self.lr=lr
        self.local_ep=local_ep
        self.dataset_train=DataLoader(dataset_train,batch_size=batch_size, shuffle=True)
        self.dataset_val=DataLoader(dataset_val, batch_size=val_batch_size, shuffle=True)
        # clients[idx]=self
    
    def train(self,net):
        self.loss_func=HingeLoss(net)
        net.train()
        optimizer=torch.optim.SGD(net.parameters(),lr=self.lr)
        
        epoch_acc=[]
        epoch_loss=[]
        for iter in range(self.local_ep):
            batch_acc=[]
            batch_loss=[]
            for batch_idx, (data,labels) in enumerate(self.dataset_train):

                labels=labels.to(torch.float)
                
                data,labels=data.to(self.device), labels.to(self.device)
                

                optimizer.zero_grad()
                data=data.to(torch.float)
                fx=net(data)
                preds=torch.where(fx>0, torch.tensor(1.), torch.tensor(-1))
                # print("fx :\n ",fx)
                # print("preds: \n",preds)
                loss=self.loss_func(fx,labels)
                acc=calculate_accuracy(preds,labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

            prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(self.idx,iter,acc.item(),loss.item()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            epoch_acc.append(sum(batch_acc)/len(batch_acc))


        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), sum(epoch_acc)/len(epoch_acc)
    
    def validate(self,net,ell):
        global targets, outputs
        net.eval()

        epoch_acc=[]
        epoch_loss=[]
        with torch.no_grad():
            batch_acc=[]
            batch_loss=[]
            for batch_idx, (data,labels) in enumerate(self.dataset_val):

                labels=labels.to(torch.float)
                data,labels=data.to(self.device), labels.to(self.device)
                data=data.to(torch.float)
                fx=net(data)
                preds=torch.where(fx>0, torch.tensor(1.), torch.tensor(-1))
                outputs.extend(preds.cpu().detach().numpy().tolist())

                targets.extend(labels.cpu().detach().numpy().tolist())

                loss=self.loss_func(fx, labels)
                acc=calculate_accuracy(preds,labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

            prGreen('Client{} Val => \tLoss: {:.4f}\tAcc: {:.3f}'.format(self.idx,loss.item(),acc.item()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            epoch_acc.append(sum(batch_acc)/len(batch_acc))

            clr=classification_report(np.array(targets),np.array(outputs),output_dict=True)

            # curr_f1= (clr[str(idx)]['f1-score']+clr[str((idx+1)%10)]['f1-score'])/2

            # macro_avg_f1_3classes.append(curr_f1)
            # macro_avg_f1_dict[idx]=curr_f1
            targets=[]
            outputs=[]

            unique_val.append(sum(batch_acc)/len(batch_acc))
        return sum(epoch_loss)/len(epoch_loss), sum(epoch_acc)/len(epoch_acc)



    def FedAvg(w):
        w_avg=copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1,len(w)):
                w_avg[k]+=w[i][k]
            w_avg[k]=torch.div(w_avg[k],len(w))
        return w_avg


if __name__=="__main__":
    program="Vehicle_FL"
    unique_val=[]
    
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    device=torch.device('cpu')

    args=parse_arguments()

    SEED=args.seed
    epochs=args.epochs
    save_model=args.save_model
    # frac=args.fac
    lr=args.lr
    
    global outputs, targets
    targets=[]
    outputs=[]

    global clients
    clients={}

    mode="online"
    if args.disable_wandb:
        mode="disabled"
    
    wandb.init(entity="prarabdh-10", project="pfsl", mode = mode)
    # wandb.run.name = args.opt_iden
    config = wandb.config          
    config.batch_size = args.batch_size    
    config.test_batch_size = args.test_batch_size        
    config.epochs = args.epochs             
    config.lr = args.lr       
    # config.dataset = args.dataset
    # config.model = args.model
    config.seed = args.seed
    # config.opt = args.opt_iden

    def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
    def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))    


    vehicle_dataset=load_vehicle()
    num_clients=len(vehicle_dataset)
    # num_clients=1
    input_shape=vehicle_dataset[0].X.shape[1]
    net_glob=LinearModel(input_shape)
    net_glob.to(device)
    print(net_glob)

    net_glob.train()

    w_glob=net_glob.state_dict()

    loss_train_collect = []
    acc_train_collect = []
    loss_val_collect = []
    acc_val_collect = []
    macro_avg_f1_3classes=[]
    macro_avg_f1_dict={}
    all_clients_val_acc=dict()
    all_clients_train_acc=dict()

    for idx in range(num_clients):
        all_clients_val_acc[idx]=[]
        all_clients_train_acc[idx]=[]
    
    st=time.time()
    max_epoch, max_f1, max_accuracy=0,0,0

    for iter in range(epochs):
        w_locals, loss_locals_train, acc_locals_train, loss_locals_val, acc_locals_val, macro_avg_f1_3classes  =[], [], [], [], [], []

        for idx in range(num_clients):
            
            full_dataset= vehicle_dataset[idx]
            dataset_train,dataset_val,dataset_test=val_test_split(full_dataset,seed=args.seed)


            local=LocalUpdate(idx,lr,device,args.local_ep,args.batch_size,args.val_batch_size,dataset_train, dataset_val)

            w,loss_train,acc_train= local.train(net=copy.deepcopy(net_glob).to(device))

            w_locals.append(copy.deepcopy(w))

            loss_locals_train.append(copy.deepcopy(loss_train))

            acc_locals_train.append(copy.deepcopy(acc_train))

            all_clients_train_acc[idx].append(acc_train)

            loss_val, acc_val = local.validate(net = copy.deepcopy(net_glob).to(device), ell=iter)
            loss_locals_val.append(copy.deepcopy(loss_val))
            acc_locals_val.append(copy.deepcopy(acc_val))
            all_clients_val_acc[idx].append(acc_val)
        
        w_glob = FedAvgM(w_locals)
        print("------------------------------------------------")
        print("------ Federation process at Server-Side -------")
        print("------------------------------------------------")

        net_glob.load_state_dict(w_glob)

        acc_avg_train = sum(acc_locals_train) / len(acc_locals_train)
        acc_train_collect.append(acc_avg_train)
        acc_avg_val = sum(acc_locals_val) / len(acc_locals_val)
        acc_val_collect.append(acc_avg_val)
    

        # f1_avg_all_user=sum(macro_avg_f1_3classes)/ len(macro_avg_f1_3classes)

        if(acc_avg_val> max_accuracy):
            max_accuracy=acc_avg_val
            max_epoch=iter
            print("Max Accuracy: ", max_accuracy) 
        loss_avg_train = sum(loss_locals_train) / len(loss_locals_train)
        loss_train_collect.append(loss_avg_train)
        loss_avg_val = sum(loss_locals_val) / len(loss_locals_val)
        loss_val_collect.append(loss_avg_val)

        print('------------------- SERVER ----------------------------------------------')
        print('Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(iter, acc_avg_train, loss_avg_train))
        print('Val:  Round {:3d}, Avg Accuracy {:.3f} |  Avg Loss {:.3f}'.format(iter, acc_avg_val, loss_avg_val))
        # if(args.setting=='setting2'):
        # print('Avg F1 Score{:.3f}'.format( f1_avg_all_user ))
        # print('-------------------------------------------------------------------------')

    print("Training and Evaluation completed!")    
    et = time.time()
    print(f"Total time taken is {(et-st)/60} mins")
    print("Max validation accuracy of unique client is: ", max(unique_val))

    round_process = [i for i in range(1, len(acc_train_collect)+1)]
    df = DataFrame({'round': round_process,'acc_train':acc_train_collect, 'acc_val':acc_val_collect})     
    file_name = f"results/FL/{program}_{args.batch_size}_vehicle_{args.lr}_{args.epochs}_base"+".xlsx"  
    df.to_excel(file_name, sheet_name= "v1_val", index = False) 

    X = range(args.epochs)
    all_clients_stacked_val  = np.array([val_acc for _,val_acc in all_clients_val_acc.items()])
    all_clients_stacked_train = np.array([train_acc for _,train_acc in all_clients_train_acc.items()])

    epochs_train_std = np.std(all_clients_stacked_train,axis = 0, dtype = np.float64)
    epochs_val_std = np.std(all_clients_stacked_val,axis = 0, dtype = np.float64)


    Y_train = acc_train_collect
    Y_train_lower = Y_train - (1.65 * epochs_train_std) #95% of the values lie between 1.65*std
    Y_train_upper = Y_train + (1.65 * epochs_train_std)

    Y_val = acc_val_collect
    Y_val_lower = Y_val - (1.65 * epochs_val_std) #95% of the values lie between 1.65*std
    Y_val_upper = Y_val + (1.65 * epochs_val_std)

    Y_train_cv =  epochs_train_std / Y_train
    Y_val_cv = epochs_val_std / Y_val

    plt.figure(0)
    plt.plot(X, Y_train)
    plt.fill_between(X,Y_train_lower , Y_train_upper, color='blue', alpha=0.25)
    plt.savefig(f'./results/train_acc_vs_epoch/vehicle_{num_clients}clients_{args.epochs}epochs_{args.batch_size}batch_base66.png', bbox_inches='tight')
   
    plt.show()
    wandb.log({"train_plot": wandb.Image(plt)})

    plt.figure(1)
    plt.plot(X, Y_val)
    plt.fill_between(X,Y_val_lower , Y_val_upper, color='blue', alpha=0.25)
    plt.savefig(f'./results/val_acc_vs_epoch/vehicle_{num_clients}clients_{args.epochs}epochs_{args.batch_size}batch_base.png', bbox_inches='tight')
    plt.show()
    wandb.log({"val_plot": wandb.Image(plt)})

    plt.figure(2)
    plt.plot(X, Y_train_cv)
    plt.show()
    wandb.log({"train_cv": wandb.Image(plt)})

    plt.figure(3)
    plt.plot(X, Y_val_cv)
    plt.show()
    wandb.log({"val_cv": wandb.Image(plt)})

    if save_model:
        torch.save(net_glob.state_dict(),f'./saved_models/model{time.time()}.pt')

    #=============================================================================
    #                         Program Completed
    #============================================================================= 