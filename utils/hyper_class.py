from pathlib import Path
import json
import os


class HyperClass(object):
    r"""The class to store and handle all the hyper parameters

    Should be sent into main_func
    """

    def __init__(self,
                 seed=20124861,
                 num_epochs=100,
                 batch_size=128,
                 learning_rate=0.01,
                 num_workers=4,
                 optimizer="sgd",
                 exp_path_name="cnn2",
                 base_path="/mnt/emc01/zeyu/mlcw/exp/",
                 network_name="resnet-18"
                 ):
        self.network_name = network_name
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.exp_path_name = exp_path_name
        self.base_path = base_path
        self.exp_path = str(Path(base_path).joinpath(exp_path_name))
        # create dir for experiments
        os.makedirs(self.exp_path, exist_ok=True)

    def save(self):
        data = self.__dict__
        json.dump(data, open(Path(self.exp_path).joinpath("hyper.json"), "w"))
