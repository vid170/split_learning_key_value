import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


def plot_class_distribution(clients, dataset, batch_size, epochs, opt_iden, client_ids, train_flag):
    class_distribution=dict()
    number_of_clients=len(client_ids)

    if(len(client_ids)<=20):
        plot_for_clients=client_ids
    else:
        plot_for_clients=random.sample(client_ids, 20)
    
    fig, ax = plt.subplots(nrows=(int(math.ceil(len(plot_for_clients)/5))), ncols=5, figsize=(15, 10))
    j=0
    i=0

    #plot histogram
    for client_id in plot_for_clients:
        if(train_flag):
            df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
        else:
            df=pd.DataFrame(list(clients[client_id].test_dataset), columns=['images', 'labels'])
        class_distribution[client_id]=df['labels'].value_counts().sort_index()
        df['labels'].value_counts().sort_index().plot(ax = ax[i,j], kind = 'bar', ylabel = 'frequency', xlabel=client_id)
        j+=1
        if(j==5 or j==10 or j==15):
            i+=1
            j=0
    fig.tight_layout()
    plt.show()
    # wandb.log({"Histogram": wandb.Image(plt)})
    
    if(train_flag):
        plt.savefig(f'plot_cifar10_data_heterogenity_100_clients_train_{opt_iden}.png')
    else:
        plt.savefig(f'plot_cifar10_data_heterogenity_100_clients_test_{opt_iden}.png')
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_histogram.png')  

    max_len=0
    #plot line graphs
    # for client_id in plot_for_clients:
    #     df=pd.DataFrame(list(clients[client_id].train_dataset), columns=['images', 'labels'])
    #     df['labels'].value_counts().sort_index().plot(kind = 'line', ylabel = 'frequency', label=client_id)
    #     max_len=max(max_len, list(df['labels'].value_counts(sort=False)[df.labels.mode()])[0])
    # plt.xticks(np.arange(0,10))
    # plt.ylim(0, max_len)
    # # plt.legend()
    # plt.show()
    # plt.savefig('plot_setting1_key_value_line_graph')
    # wandb.log({"Line graph": wandb.Image(plt)})
    # plt.savefig(f'./results/class_vs_freq/{dataset}_{number_of_clients}clients_{epochs}epochs_{batch_size}batch_{opt}_line_graph.png')
    
    return plot_for_clients

def plot_client_frequency(clients, opt_iden):
    y=[]
    for client_id, client in clients.items():
        y.append(len(client.train_dataset))
    x=np.array([i for i in range(len(clients))])
    y=np.array(y)
    print(x)
    print(y)
    plt.clf()
    plt.plot(x,y)
    plt.show()
    plt.savefig(f'results/client_vs_frequency_{opt_iden}.png')

