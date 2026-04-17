from torch_geometric.data import InMemoryDataset, TemporalData, download_url
import torch
import os.path as osp
from typing import Callable, Optional
import networkx as nx
import pandas as pd


class ECHRData(InMemoryDataset):

    def __init__(
        self,
        root: str,
        name:str = 'echr',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        
        self.pre_process()
        
        self.name = name.lower()

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        
        self.load(self.processed_paths[0], data_cls=TemporalData)

    def pre_process(self) -> None:
        nodes = pd.read_csv('/users/sgdbareh/volatile/ECHR_Importance/Knowledge_Graph/network_nodes.csv')
        edges = pd.read_csv('/users/sgdbareh/volatile/ECHR_Importance/Knowledge_Graph/network_edges.csv')

        df = pd.merge(edges, nodes, left_on='source', right_on='id')
        edge = df[['source', 'target', 'date']]
        reference_date = pd.Timestamp('1996-12-18')
        edge['date'] = (pd.to_datetime(edge['date']) - reference_date).dt.days

        sorted_edges = edge.sort_values(by='date')

        # Create a mapping from filenames to integers
        unique_filenames = pd.concat([sorted_edges['source'], sorted_edges['target']]).unique()
        self.filename_to_int = {filename: idx + 1 for idx, filename in enumerate(unique_filenames)}

        sorted_edges['source'] = sorted_edges['source'].map(self.filename_to_int)
        sorted_edges['target'] = sorted_edges['target'].map(self.filename_to_int)

        nodes.drop(['outcome', 'appno', 'ecli', 'facts', 'the_law','date'], axis=1, inplace=True)
        nodes['id'] = nodes['id'].map(self.filename_to_int)
        nodes = nodes.dropna(subset=['id'])
        nodes['id'] = nodes['id'].astype(int)

        #df = df.drop_duplicates(subset='source')

        categories=[1,2,3,4]
        nodes['importance'] = pd.Categorical(nodes['importance'], categories=categories, ordered=True)
        nodes['importance'] = nodes['importance'].cat.codes

        categories_branch = ['ADMISSIBILITY','ADMISSIBILITYCOM','COMMITTEE','CHAMBER','DECGRANDCHAMBER','GRANDCHAMBER']
        nodes['branch'] = pd.Categorical(nodes['branch'], categories=categories_branch, ordered=True)
        nodes['branch'] = nodes['branch'].cat.codes
        nodes['branch'] = nodes['branch'] + 1

        df_2 = nodes['respondent'].str.get_dummies(sep=';')
        nodes = pd.concat([nodes, df_2], axis=1)
        nodes.drop(columns=['respondent'], inplace=True)

        # Convert the DataFrame to a numpy array and then to a PyTorch tensor
        df_tensor = torch.tensor(nodes.values[:,1:], dtype=torch.float32)

        self.df_tensor = df_tensor
        self.sorted_edges = sorted_edges

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.csv'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self) -> None:
        
        edges = self.sorted_edges

        src = torch.from_numpy(edges.iloc[:, 0].values).to(torch.long)
        dst = torch.from_numpy(edges.iloc[:, 1].values).to(torch.long)
        t = torch.from_numpy(edges.iloc[:, 2].values).to(torch.long)
        y = torch.zeros(edges.shape[0], dtype=torch.long)
        msg = torch.zeros((edges.shape[0], 5), dtype=torch.float)
        x = self.df_tensor

        data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y, x=x)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])