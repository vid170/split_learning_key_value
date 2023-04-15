import os
import torch
import torch.nn.functional as F
import multiprocessing
from threading import Thread
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
from utils.split_dataset import DatasetFromSubset
import pickle
import queue
import struct
import numpy as np
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss


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


class Client(Thread):
    def __init__(self, id, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.id = id
        self.flag=0
        self.test_flag=0
        self.front_model = []
        self.back_model = []
        self.losses = []
        self.current_keys=[]
        self.all_keys=[]
        self.target_mappings={}
        self.activation_mappings={}
        self.data_key=0
        self.test_target_mappings={}
        self.test_activation_mappings={}
        self.test_data_key=0
        self.train_dataset = None
        self.test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.iterator = None
        self.test_iterator=None
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None
        self.criterion = None
        self.data = None
        self.targets = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.train_acc = []
        self.test_acc = []
        self.front_epsilons = []
        self.front_best_alphas = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device
        # self.device = torch.device('cpu')


    # def softmax_helper(x):
    #     """This function computes the softmax using torch functionnal on the 1-axis.
    #     Parameters
    #     ----------
    #     x : torch.Tensor
    #         The input.
    #     Returns
    #     -------
    #     torch.Tensor
    #         Output
    #     """
    #     return F.softmax(x, 1)

    def Dice_coef(output2, target, eps=1e-5):  # dice score used for evaluation
        target = target.float()
        # print(type(output2), "====", type(target))
        output2 = output2.float()
        num = 2 * (output2 * target).sum()
        den = output2.sum() + target.sum() + eps
        return num / den, den, num

    def metric(predictions, gt):
        gt = gt.float()
        predictions = predictions.float()
        # Compute tumor+kidney Dice >0 (1+2)
        tk_pd = torch.gt(predictions, 0)
        tk_gt = torch.gt(gt, 0)
        tk_dice, denom, num = Dice_coef(tk_pd.float(), tk_gt.float())  # Composite
        tu_dice, denom, num = Dice_coef((predictions == 2).float(), (gt == 2).float())

        return (tk_dice + tu_dice) / 2



    def backward_back(self):
        self.loss.backward()

    def set_targets(self):
        self.targets=torch.Tensor(np.array([self.target_mappings[x] for x in self.current_keys])).type(torch.int64).to(self.device)

    def set_test_targets(self):
        self.targets=torch.Tensor(np.array([self.test_target_mappings[x] for x in self.current_keys])).type(torch.int64).to(self.device)
           
    def backward_front(self):
        self.activations1.backward(self.remote_activations1.grad)

    

    def calculate_loss(self):
        bloss=BaselineLoss()


        self.loss=bloss(self.outputs, self.targets)
        # self.loss.backward()

    def calculate_test_acc(self):

        with torch.no_grad():
             
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.test_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.test_acc[-1]}')

    def calculate_test_acc_kits(self):
        
        y_pred = self.outputs.detach().cpu()
        # print(y_pred.shape, "---", type(y_pred))
        preds_softmax = F.softmax(y_pred, 1)
        # F.softmax(x, 1)
        preds = preds_softmax.argmax(1)
        y =     self.targets.detach().cpu()
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


    def calculate_train_acc(self):
        with torch.no_grad():
            _, self.predicted = torch.max(self.outputs.data, 1)
            self.n_correct = (self.predicted == self.targets).sum().item()
            self.n_samples = self.targets.size(0)
            # self.train_acc.append(100.0 * self.n_correct/self.n_samples)
            return 100.0 * self.n_correct/self.n_samples
            # print(f'Acc: {self.train_acc[-1]}')

    
    def calculate_train_acc_kits(self):
        y_pred = self.outputs.detach().cpu()
        # print(y_pred.shape, "---", type(y_pred))
        preds_softmax = F.softmax(y_pred, 1)
        # F.softmax(x, 1)
        preds = preds_softmax.argmax(1)
        y =     self.targets.detach().cpu()
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




    def connect_server(self, host='localhost', port=8000, BUFFER_SIZE=4096):
        self.socket, self.server_socket = multiprocessing.Pipe()
        print(f"[*] Client {self.id} connecting to {host}")

    def create_DataLoader(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_DataLoader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                batch_size=self.train_batch_size,
                                                shuffle=True)
        self.test_DataLoader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                batch_size=self.test_batch_size,
                                                shuffle=True)

    def disconnect_server(self) -> bool:
        if not is_socket_closed(self.socket):
            self.socket.close()
            return True
        else:
            return False


    def forward_back(self):
        self.back_model.to(self.device)
        self.outputs = self.back_model(self.remote_activations2)
        # print("---", self.outputs)
        # print("+++", self.outputs.shape)

    def forward_front(self):
        self.data, self.targets = next(self.iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)

        self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)

        self.remote_activations1 = self.activations1.detach().requires_grad_(True)

        # return self.activations1

    
    def forward_front_key_value(self):
        self.data, self.targets = next(self.iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
        self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        local_activations1=list(self.activations1.cpu().detach().numpy())
        local_targets=list(self.targets.cpu().detach().numpy().astype(int))
        
        for i in range(0,len(local_targets)):
            self.activation_mappings[self.data_key]=local_activations1[i]
            self.target_mappings[self.data_key]=local_targets[i]
            self.data_key+=1
    
    def forward_front_test_key_value(self):
        self.data, self.targets = next(self.test_iterator)
        self.data, self.targets = self.data.to(self.device), self.targets.to(self.device)
        self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        local_activations1=list(self.activations1.cpu().detach().numpy())
        local_targets=list(self.targets.cpu().detach().numpy().astype(int))
        
        for i in range(0,len(local_targets)):
            self.test_activation_mappings[self.test_data_key]=local_activations1[i]
            self.test_target_mappings[self.test_data_key]=local_targets[i]
            self.test_data_key+=1
        


    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']


    def get_remote_activations1_grads(self):
        self.remote_activations1.grad = get_object(self.socket)


    def get_remote_activations2(self):
        self.remote_activations2 = get_object(self.socket)


    def idle(self):
        pass


    def load_data(self, dataset, transform):
        try:
            dataset_path = os.path.join(f'data/{dataset}/{self.id}')
        except:
            raise Exception(f'Dataset not found for client {self.id}')
        self.train_dataset = torch.load(f'{dataset_path}/train/{self.id}.pt')
        self.test_dataset = torch.load(f'{dataset_path}/test/{self.id}.pt')

        self.train_dataset = DatasetFromSubset(
            self.train_dataset, transform=transform
        )
        self.test_dataset = DatasetFromSubset(
            self.test_dataset, transform=transform
        )


    # def onThread(self, function, *args, **kwargs):
    #     self.q.put((function, args, kwargs))


    # def run(self, *args, **kwargs):
    #     super(Client, self).run(*args, **kwargs)
    #     while True:
    #         try:
    #             function, args, kwargs = self.q.get(timeout=self.timeout)
    #             function(*args, **kwargs)
    #         except queue.Empty:
    #             self.idle()


    def send_remote_activations1(self):
        send_object(self.socket, self.remote_activations1)
    

    def send_remote_activations2_grads(self):
        send_object(self.socket, self.remote_activations2.grad)


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()


    # def train_model(self):
    #     forward_front_model()
    #     send_activations_to_server()
    #     forward_back_model()
    #     loss_calculation()
    #     backward_back_model()
    #     send_gradients_to_server()
    #     backward_front_model()


    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()

