from threading import Thread
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
import pickle
import queue
import struct
import torch


def handle(client, addr, file):
    buffsize = 1024
    # file = '/home/ashutosh/score_report.pdf'
    # print('File size:', os.path.getsize(file))
    fsize = struct.pack('!I', len(file))
    print('Len of file size struct:', len(fsize))
    client.send(fsize)
    # with open(file, 'rb') as fd:
    while True:
        chunk = fd.read(buffsize)
        if not chunk:
            break
        client.send(chunk)
    fd.seek(0)
    hash = hashlib.sha512()
    while True:
        chunk = fd.read(buffsize)
        if not chunk:
            break
        hash.update(chunk)
    client.send(hash.digest())


class ConnectedClient(object):
    # def __init__(self, id, conn, address, loop_time=1/60, *args, **kwargs):
    def __init__(self, id, conn, *args, **kwargs):
        super(ConnectedClient, self).__init__(*args, **kwargs)
        self.id = id
        self.conn = conn
        self.front_model = None
        self.back_model = None
        # self.center_model = None
        self.center_front_model=None
        self.center_back_model=None
        self.train_fun = None
        self.test_fun = None
        self.keepRunning = True
        self.current_keys=[]   
        self.a1 = None
        self.a2 = None
        self.flag=0
        self.center_optimizer = None
        self.activation_mappings={}
        self.test_activation_mappings={}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # def onThread(self, function, *args, **kwargs):
    #     self.q.put((function, args, kwargs))


    # def run(self, loop_time=1/60, *args, **kwargs):
    #     super(ConnectedClient, self).run(*args, **kwargs)
    #     while True:
    #         try:
    #             function, args, kwargs = self.q.get(timeout=self.timeout)
    #             function(*args, **kwargs)
    #         except queue.Empty:
    #             self.idle()

    
    def forward_center(self):
        self.activations2 = self.center_model(self.remote_activations1)
        self.remote_activations2 = self.activations2.detach().requires_grad_(True)
    def forward_center_front(self):
        self.middle_activations=self.center_front_model(self.remote_activations1)
        local_middle_activations=list(self.middle_activations.cpu().detach().numpy())
        for i in range(0, len(self.current_keys)):
            # if(test==True):

            #     self.test_activation_mappings[self.current_keys[i]]=local_middle_activations[i]
            # else:
            self.activation_mappings[self.current_keys[i]]=local_middle_activations[i]
    
    def forward_center_front_test(self):
        self.middle_activations=self.center_front_model(self.remote_activations1)
        local_middle_activations=list(self.middle_activations.cpu().detach().numpy())
        for i in range(0, len(self.current_keys)):
            # if(test==True):
                
            #     self.test_activation_mappings[self.current_keys[i]]=local_middle_activations[i]
            # else:
            self.test_activation_mappings[self.current_keys[i]]=local_middle_activations[i]
    
    def forward_center_back(self):
        self.activations2=self.center_back_model(self.middle_activations)
        self.remote_activations2=self.activations2.detach().requires_grad_(True)

    def backward_center(self):
        self.activations2.backward(self.remote_activations2.grad)

    def update_all_keys(self):
        self.all_keys=list(set(self.all_keys)-set(self.current_keys))

    def idle(self):
        pass


    def connect(self):
        pass


    def disconnect(self):
        if not is_socket_closed(self.conn):
            self.conn.close()
            return True
        else:
            return False


    # def _send_model(self):
    def send_model(self):
        model = {'front': self.front_model, 'back': self.back_model}
        send_object(self.conn, model)
        # handle(self.conn, self.address, model)


    # def send_optimizers(self):
    #     # This is just a sample code and NOT optimizers. Need to write code for initializing optimizers
    #     optimizers = {'front': self.front_model.parameters(), 'back': self.back_model.parameters()}
    #     send_object(self.conn, optimizers)


    def send_activations(self, activations):
        send_object(self.conn, activations)


    def get_remote_activations1(self):
        self.remote_activations1 = get_object(self.conn)


    def send_remote_activations2(self):
        send_object(self.conn, self.remote_activations2)


    def get_remote_activations2_grads(self):
        self.remote_activations2.grad = get_object(self.conn)


    def send_remote_activations1_grads(self):
        send_object(self.conn, self.remote_activations1.grad)

    # def send_model(self):
    #     self.onThread(self._send_model)        
