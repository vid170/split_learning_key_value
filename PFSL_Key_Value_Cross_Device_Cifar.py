import random
import time
import torch
from math import ceil
from torch.utils.data import  Dataset
from utils.client_simulation import generate_random_clients
from utils.arg_parser import parse_arguments
import matplotlib.pyplot as plt
import time
import torch.optim as optim 
import copy
from datetime import datetime
import numpy as np
from ConnectedClient import ConnectedClient
import importlib
from utils.merge import  merge_weights
import wandb
import time 
from utils import datasets, dataset_settings
from utils.partition import partition_data
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_min_client(clients):
    min_client_id=clients[client_ids[0]]
    min_client_len=500000

    for client_id, client in clients.items():
        curr_len=len(client.train_dataset)
        if curr_len<min_client_len:
            min_client_id=client_id 
            min_client_len=curr_len
    return min_client_id
        
    
if __name__ == "__main__":    

    
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Arguments provided", args)

    mode = "online"
    if args.disable_wandb:
        mode = "disabled"
        
    wandb.init(entity="iitbhilai", project="Split_learning exps", mode = mode)
    wandb.run.name = args.opt_iden

    config = wandb.config          
    config.batch_size = args.batch_size    
    config.test_batch_size = args.test_batch_size        
    config.epochs = args.epochs             
    config.lr = args.lr       
    config.dataset = args.dataset
    config.model = args.model
    config.seed = args.seed
    config.opt = args.opt_iden   

    max_acc=0                           


    random.seed(args.seed)
    torch.manual_seed(args.seed)

    overall_test_acc = []
    overall_train_acc = []

    print('Generating random clients...', end='')
    clients = generate_random_clients(args.number_of_clients)
    client_ids = list(clients.keys())    
    print('Done')

    input_channels=3
    print(f'Random client ids:{str(client_ids)}')
    transform=None
    max_epoch=0

    print('Initializing clients...')
       
    train_full_dataset, test_full_dataset, input_channels = datasets.load_full_dataset(args.dataset, "data", args.number_of_clients, args.datapoints, args.pretrained)
    dict_users=partition_data(train_full_dataset , 10, "noniid-labeldir", args.number_of_clients, 0.5, 1234)
    
    client_idx=0
    dict_user_train=dict()
    dict_user_test=dict()
    client_idxs=dict()
    for _, client in clients.items():
        dict_user_train[_]=dict_users[client_idx]
        client_idxs[_]=client_idx
        client_idx+=1
    for _, client in clients.items():
        client.train_dataset=DatasetSplit(train_full_dataset, dict_user_train[_])
      
   
    test_frequency_list=(np.random.dirichlet(np.ones(args.number_of_clients), size=1)*len(test_full_dataset)).tolist()
    
    idx=0
    test_dict_frequency={}
    for client_id in client_ids:
        test_dict_frequency[client_id]= test_frequency_list[0][idx]
        idx+=1
       
    dict_users2, similar_client_ids = dataset_settings.create_test_data_cross_device(clients, args.number_of_clients, 0.9, client_ids, test_dict_frequency, 10, test_full_dataset, client_idxs)
   
    print("SIMILAR CLIENTS: ", similar_client_ids)
    
    for _, client in clients.items():
        dict_user_test[_]=dict_users2[client_idxs[_]]
        print("test data alloted, its length is: ", len(dict_user_test[_]))
    for _, client in clients.items():
        client.test_dataset=DatasetSplit(test_full_dataset, dict_user_test[_])
        print("test dataset length: ", len(client.test_dataset), " train dataset length: ", len(client.train_dataset))
        client.create_DataLoader(args.batch_size, args.test_batch_size)


    print('Client Intialization complete.')

     #Setting the start of personalisation phase
    if(args.setting!='setting2'):
        args.checkpoint=args.epochs+10

    
    # plot_for_clients=plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, similar_client_ids, True)
    # plot_class_distribution(clients, args.dataset, args.batch_size, args.epochs, args.opt_iden, plot_for_clients, False)

    # plot_client_frequency(clients, args.opt_iden)


    #Assigning front, center and back models and their optimizers for all the clients
    model = importlib.import_module(f'models.{args.model}')

    for _, client in clients.items():
        client.front_model = model.front(input_channels, pretrained=args.pretrained)
        client.back_model = model.back(pretrained=args.pretrained)
    print('Done')

     

    for _, client in clients.items():
        client.front_optimizer = optim.Adam(client.front_model.parameters(), lr=args.lr)
        client.back_optimizer = optim.Adam(client.back_model.parameters(), lr=args.lr)
    for _, client in clients.items():
        client.num_iterations=ceil(len(client.train_DataLoader.dataset)/args.batch_size)
        client.num_test_iterations=ceil(len(client.test_DataLoader.dataset)/args.test_batch_size)

   
    min_client= clients[get_min_client(clients)]
   
    num_iterations = ceil(len(min_client.train_DataLoader.dataset)/args.batch_size)
    num_test_iterations = ceil(len(min_client.test_DataLoader.dataset)/args.test_batch_size)

    sc_clients = {} #server copy clients
   

    for iden in client_ids:
        sc_clients[iden] = ConnectedClient(iden, None)

    for _,s_client in sc_clients.items():
        s_client.center_front_model = model.center_front(pretrained=args.pretrained)
        s_client.center_front_model.to(device)
        s_client.center_back_model = model.center_back(pretrained=args.pretrained)
        s_client.center_back_model.to(device)
        s_client.center_optimizer = optim.Adam(s_client.center_front_model.parameters(), args.lr)


    st = time.time()

    
    for epoch in range(1):

        for _, client in clients.items():
            client.iterator = iter(client.train_DataLoader)
            client.test_iterator=iter(client.test_DataLoader)

        for _, client in clients.items():
            for iteration in range(client.num_iterations):
                client.forward_front_key_value()
        
        for _, client in clients.items():
            sc_clients[_].activation_mappings=client.activation_mappings
            sc_clients[_].all_keys = list(sc_clients[_].activation_mappings.keys())      

        for client_id, client in sc_clients.items():    
        
           
            for iteration in range(clients[client_id].num_iterations):
                print("training center layer front part: epoch ", epoch, "iteration ", iteration, "/", clients[client_id].num_iterations)
               
                client.current_keys=list(np.random.choice(client.all_keys, min(clients[client_id].train_batch_size, len(client.all_keys)), replace=False))
                client.update_all_keys()
                clients[client_id].activations1=torch.Tensor(np.array([client.activation_mappings[x] for x in client.current_keys])).to(device)
                clients[client_id].remote_activations1=clients[client_id].activations1.detach().requires_grad_(True)
                client.remote_activations1=clients[client_id].remote_activations1
                client.forward_center_front()

        for _, client in clients.items():
            for iteration in range(client.num_test_iterations):
                client.forward_front_test_key_value()

        for _, client in clients.items():
            sc_clients[_].test_activation_mappings=client.test_activation_mappings
            sc_clients[_].all_keys = list(sc_clients[_].test_activation_mappings.keys())
        
        
        for client_id, client in sc_clients.items():
            
            for iteration in range(clients[client_id].num_test_iterations):
                client.current_keys=list(np.random.choice(client.all_keys, min(clients[client_id].test_batch_size, len(client.all_keys)), replace=False))
                client.update_all_keys()
                clients[client_id].activations1=torch.Tensor(np.array([client.test_activation_mappings[x] for x in client.current_keys])).to(device)
                clients[client_id].remote_activations1=clients[client_id].activations1.detach().requires_grad_(True)
                client.remote_activations1=clients[client_id].remote_activations1
                client.forward_center_front_test()
                           
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

   
    for epoch in range(args.epochs):
       
        if(epoch==args.checkpoint): # When starting epoch of the perosnalisation is reached, freeze all the layers of the center model 
            for client_id in client_ids:
                sc_clients[client_id].center_back_model.load_state_dict(torch.load('saved_models/best_generalized_model_sc_client_center_back.pt'))
                clients[client_id].back_model.load_state_dict(torch.load('saved_models/best_generalized_model_client_back.pt'))
                if(client_id in similar_client_ids):
                    sc_clients[client_id].center_back_model.freeze(epoch, pretrained=True)

        overall_train_acc.append(0)

        for _, client in clients.items():
            client.train_acc.append(0)
            
        for _, client in sc_clients.items():
            client.all_keys = list(client.activation_mappings.keys())
            
        for iteration in range(num_iterations):
            print("training center layer back part onwards epoch ", epoch, "iteration ", iteration, "/", num_iterations)

            for client_id, client in sc_clients.items():
                client.current_keys=list(np.random.choice(client.all_keys, min(clients[client_id].train_batch_size, len(client.all_keys)), replace=False))
                client.update_all_keys()
                client.middle_activations=torch.Tensor(np.array([client.activation_mappings[x] for x in client.current_keys])).to(device).detach().requires_grad_(True)
                client.forward_center_back()
            
            for client_id, client in clients.items():
                client.remote_activations2 = sc_clients[client_id].remote_activations2
                client.forward_back()
                client.current_keys=sc_clients[client_id].current_keys
                client.set_targets()

            for _, client in clients.items():
                client.calculate_loss()
            
            for _, client in clients.items():
                client.backward_back()

            for client_id, client in sc_clients.items():
                client.remote_activations2 = clients[client_id].remote_activations2
                client.backward_center()

            for _, client in clients.items():
                client.step_back()
                client.zero_grad_back()

            
            for _, client in sc_clients.items():
                client.center_optimizer.step()
                client.center_optimizer.zero_grad()

            for _, client in clients.items():
                client.train_acc[-1] += client.calculate_train_acc()
        
        
        for c_id, client in clients.items():
            client.train_acc[-1] /= num_iterations
            overall_train_acc[-1] += client.train_acc[-1]

        overall_train_acc[-1] /= len(clients)
        print(f' Personalized Average Train Acc: {overall_train_acc[-1]}')

        # merge weights below uncomment 
        params = []
        for _, client in sc_clients.items():
            params.append(copy.deepcopy(client.center_back_model.state_dict()))
        w_glob = merge_weights(params)
        del params

        for _, client in sc_clients.items():
            client.center_back_model.load_state_dict(w_glob)

        params = []
        #In the personalisation phase merging of weights of the back layers is stopped
        if(epoch <=args.checkpoint):
            for _, client in clients.items():
                params.append(copy.deepcopy(client.back_model.state_dict()))
            w_glob_cb = merge_weights(params)
            del params
    
            for _, client in clients.items():
                client.back_model.load_state_dict(w_glob_cb)
        if(epoch > args.checkpoint):
            for client_id in client_ids:
                if(client_id not in similar_client_ids):
                    params.append(copy.deepcopy(clients[client_id].back_model.state_dict()))
            w_glob_cb = merge_weights(params)
            del params
    
            for client_id in client_ids:
                if(client_id not in similar_client_ids):
                    clients[client_id].back_model.load_state_dict(w_glob_cb)

        client_test_acc={}

        if epoch%1 == 0:
          
            with torch.no_grad():
                test_acc = 0
                overall_test_acc.append(0)

                for _, client in clients.items():
                    client.test_acc.append(0)
                    client.pred=[]
                    client.y=[]

                for _, client in sc_clients.items():
                    client.all_keys = list(client.test_activation_mappings.keys())
                    
                for client_id, client in sc_clients.items():
                    for iteration in range(clients[client_id].num_test_iterations):
                        client.current_keys=list(np.random.choice(client.all_keys, min(clients[client_id].test_batch_size, len(client.all_keys)), replace=False))
                        client.update_all_keys()
                        client.middle_activations=torch.Tensor(np.array([client.test_activation_mappings[x] for x in client.current_keys])).to(device)
                        client.middle_activations=client.middle_activations.detach().requires_grad_(True)                       
                        client.forward_center_back()
                        clients[client_id].remote_activations2 = sc_clients[client_id].remote_activations2
                        clients[client_id].forward_back()
                        clients[client_id].current_keys=sc_clients[client_id].current_keys
                        clients[client_id].set_test_targets()
                        clients[client_id].test_acc[-1] += clients[client_id].calculate_test_acc()
                    clients[client_id].test_acc[-1]/=clients[client_id].num_test_iterations

                
                    curr_client_test_acc=clients[client_id].test_acc[-1]
                    client_test_acc[client_id]=curr_client_test_acc
                    overall_test_acc[-1] += curr_client_test_acc
                    
                  
                overall_test_acc[-1] /= len(clients)
                print("client test acc at this epoch: ", client_test_acc)
          
                print(f' Personalized Average Test Acc: {overall_test_acc[-1]}   ')
                max_acc=max(max_acc, overall_test_acc[-1])
                print("Maximum test acc: ", max_acc)
                if(max_acc==overall_test_acc[-1] and epoch<=args.checkpoint):
                    all_clients_acc_max=client_test_acc
                    for _, s_client in sc_clients.items():
                        torch.save(s_client.center_back_model.state_dict(), 'saved_models/best_generalized_model_sc_client_center_back.pt')
                        torch.save(clients[client_id].back_model.state_dict(),'saved_models/best_generalized_model_client_back.pt'  )
                clients_hurt_count=0
                clients_hurt=[]
                if(epoch>=args.checkpoint):
                    for _, client in clients.items():
                        if(client_test_acc[_]<all_clients_acc_max[_]):
                            clients_hurt_count+=1
                            clients_hurt.append(_)
                    print("clients hurt percentage: ", clients_hurt_count)
                    print("hurt clients accuracies: ", )
                    for client_id in clients_hurt:
                        print("previous accuracy: ", all_clients_acc_max[client_id], " new accuracy: ", client_test_acc[client_id])
                if(max_acc==overall_test_acc[-1]):
                    print("max accuracy achieved in personalisation phase: ", max_acc)



            wandb.log({
                "Epoch": epoch,
                "Personalized Average Train Accuracy": overall_train_acc[-1],
                "Personalized Average Test Accuracy": overall_test_acc[-1],  
            })




    timestamp = int(datetime.now().timestamp())
    plot_config = f'''dataset: {args.dataset},
                    model: {args.model},
                    batch_size: {args.batch_size}, lr: {args.lr},
                    '''

    et = time.time()
    print("all_clients_acc_max: ", all_clients_acc_max)
    # x=np.array([i for i in range(args.number_of_clients)])
    # plt.clf()
    # plt.plot(x, all_clients_acc_max)
    # plt.show()
    # plt.savefig('all_clients_max_acc.png')
    print(f"Time taken for this run {(et - st)/60} mins")
    wandb.log({"time taken by program in mins": (et - st)/60})
   
