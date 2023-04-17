import pandas as pd
from monai.data.dataset import Dataset, CacheDataset
from pathlib import Path

class KITSDataset:
    def __init__(self,):
        self.thresholded_sites_path = Path('FLamby/flamby/datasets/fed_kits19/metadata/thresholded_sites.csv')
        self.thresholded_sites = pd.read_csv(self.thresholded_sites_path)

    def get_client_cases(self,client_id):
        client = self.thresholded_sites.query(f'site_ids == {client_id}').reset_index(drop=True)
        train_cases = client.query(f"train_test_split == 'train'")['case_ids'].to_list()
        valid_cases = client.query(f"train_test_split == 'test'")['case_ids'].to_list()
        return train_cases, valid_cases
    
    def get_data_dict(self,client_id):
        train_cases, valid_cases = self.get_client_cases(client_id)


if __name__ == '__main__':
    pass