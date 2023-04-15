import torch 
import torch.nn 
import numpy as np
from model import LinearModel
from hinge_loss import HingeLoss
from torch.utils.data import DataLoader
from main import LocalUpdate, calculate_accuracy,val_test_split
from data import Vehicle,load_vehicle
from sklearn.metrics import classification_report
import argparse
import time
import wandb
import copy
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt



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
    parser.add_argument(
        "--model_file",
        type=str,
        default=None,
        help="Name of the saved model file without the .pt extension"
    )
    args = parser.parse_args()
    args = parser.parse_args()
    return args

class LocalTest(LocalUpdate):
    def __init__(self, idx,lr,device,local_ep,batch_size,val_batch_size,dataset_train,dataset_val,dataset_test=None):
        super(LocalTest,self).__init__(idx,lr,device,local_ep,batch_size,val_batch_size,dataset_train,dataset_val)
        self.dataset_test=dataset_test
    
    def test(self, net,ell):
        global targets, outputs
        self.loss_func=HingeLoss(net)
        net.eval()

        epoch_acc=[]
        epoch_loss=[]
        with torch.no_grad():
            batch_acc=[]
            batch_loss=[]
            for batch_idx, (data,labels) in enumerate(self.dataset_train):
                labels=labels.to(torch.float)
                data,labels=data.to(self.device), labels.to(self.device)

                data=data.to(torch.float)
                fx=net(data)

                preds=torch.where(fx>0, torch.tensor(1.),torch.tensor(-1))
                outputs.extend(preds.cpu().detach().numpy().tolist())

                targets.extend(labels.cpu().detach().numpy().tolist())

                loss=self.loss_func(fx,labels)
                acc=calculate_accuracy(preds,labels)
                batch_loss.append(loss.item())
                batch_acc.append(acc.item())

            prBlue('Client{} Test => \tLoss: {:.4f}\tAcc: {:.3f}'.format(self.idx, loss.item(),acc.item()))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            epoch_acc.append(sum(batch_acc)/len(batch_acc))

            clr=classification_report(np.array(targets),np.array(outputs),output_dict=True)

            targets=[]
            outputs=[]

            # unique_val.append(sum(batch_acc)/len(batch_acc))
        
        return sum(epoch_loss)/len(epoch_loss),sum(epoch_acc)/len(epoch_acc)



if __name__=="__main__":
    program="Vehicle_FL_test"
    unqiue_val=[]

    device=torch.device('cpu')
    args=parse_arguments()
    SEED=args.seed
    epochs=args.epochs
    save_model=args.save_model
    model_file=args.model_file
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

     
    def prBlue(skk): print("\033[94m {}\033[00m" .format(skk))    

    vehicle_dataset=load_vehicle()
    num_clients=len(vehicle_dataset)
    input_shape=vehicle_dataset[0].X.shape[1]
    net_glob=LinearModel(input_shape)
    net_glob.to(device)
    print(net_glob)
    net_glob.load_state_dict(torch.load('./saved_models/'+model_file+'.pt'))

    net_glob.eval()
    w_glob=net_glob.state_dict()
    loss_test_collect=[]
    acc_test_collect=[]
    all_clients_test_acc=dict()
    
    for idx in range(num_clients):
        all_clients_test_acc[idx]=[]
    
    st=time.time()
    max_epoch,max_f1,max_accuracy=0,0,0

    for iter in range(epochs):
        loss_locals_test,acc_locals_test=[],[]

        for idx in range(num_clients):
            full_dataset=vehicle_dataset[idx]
            dataset_train,dataset_val,dataset_test=val_test_split(full_dataset,seed=args.seed)

            local_test=LocalTest(idx,lr,device,args.local_ep,args.batch_size,args.val_batch_size,dataset_train, dataset_val, dataset_test)

            loss_test,acc_test=local_test.test(net=copy.deepcopy(net_glob).to(device), ell=iter)

            loss_locals_test.append(copy.deepcopy(loss_test))

            acc_locals_test.append(copy.deepcopy(acc_test))

            all_clients_test_acc[idx].append(acc_test)

        acc_avg_test=sum(acc_locals_test)/len(acc_locals_test)
        acc_test_collect.append(acc_avg_test)


        if(acc_avg_test>max_accuracy):
            max_accuracy=acc_avg_test
            max_epoch=iter
            print("Max Accuracy: ",max_accuracy)
        loss_avg_test=sum(loss_locals_test)/len(loss_locals_test)
        loss_test_collect.append(loss_avg_test)

        print('------------------- SERVER ----------------------------------------------')
        print('Test:  Round {:3d}, Avg Accuracy {:.3f} |  Avg Loss {:.3f}'.format(iter, acc_avg_test, loss_avg_test))

        round_process=[i for i in range(1,len(acc_test_collect)+1)]
        
        df=DataFrame({'round':round_process,'acc_test':acc_test_collect})

        file_name=f"results/tests/{program}_{args.batch_size}_vehicle_{args.lr}_{args.epochs}_base"+".xlsx"

        df.to_excel(file_name, sheet_name="v1_test",index=False)

        X=range(args.epochs)
        all_clients_stacked_test= np.array([test_acc for _,test_acc in all_clients_test_acc.items()])

        epochs_test_std=np.std(all_clients_stacked_test,axis=0,dtype=np.float64)

        Y_test=acc_test_collect
        Y_test_lower=Y_test-(1.65*epochs_test_std)
        # 95% confidence interval
        Y_test_upper= Y_test+(1.65*epochs_test_std)

        Y_test_cv=epochs_test_std/Y_test

        plt.figure(0)
        plt.plot(X,Y_test)
        plt.fill_between(X,Y_test_lower,Y_test_upper,color='blue',alpha=0.25)

        plt.savefig(f'./results/test_acc_vs_epoch/vehicle_{num_clients}clients_{args.epochs}epochs_{args.batch_size}batch_base.png',bbox_inches='tight')

        plt.show()
        wandb.log({"test_plot": wandb.Image(plt)})
        
        plt.figure(1)
        plt.plot(X,Y_test_cv)
        plt.show()
        wandb.log({"test_cv":wandb.Image(plt)})
