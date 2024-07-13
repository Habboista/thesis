import json
import os

import matplotlib.pyplot as plt

class Info:
    """Store and save training and validation information for each epoch.
    
    It allows to save and load information for each epoch and plotting.
    Each epoch is a dictionary of information.
    """
    def __init__(self, dir_path: str, periodic_plot: bool=False, period: int=10) -> None:
        self.dir_path = dir_path
        self.periodic_plot = periodic_plot
        self.period = period
        self.plot_counter = 0
        self.info: list[dict[str, list[float]]] = []
    
    def new_epoch(self) -> None:
        """Start a new epoch."""
        self.info.append(dict())

    def log_info(self, d: dict[str, float]) -> None:
        """Log information for the current epoch."""
        for k, v in d.items():
            try:
                self.info[-1][k].append(v)
            except KeyError:
                self.info[-1][k] = [v]
        
        if self.periodic_plot:
            self.plot_counter += 1
            if self.plot_counter == self.period:
                self.plot_counter = 0
                self.plot()

    def plot(self) -> None:
        for k, v in self.info[-1].items():
            plt.plot(v)
            plt.title(k)
            plt.show()
            
    def average_epoch(self, epoch:int, key: str) -> float:
        if len(self.info[epoch][key]) == 0:
            return 0.
        return sum(self.info[epoch][key]) / len(self.info[epoch][key])

    def save_epoch(self, epoch: int, filename: str) -> None:
        path = os.path.join(self.dir_path, filename)
        with open(path, 'w') as f:
            json.dump(self.info[epoch], f, indent=4)

    def load_epoch(self, filename: str) -> None:
        path = os.path.join(self.dir_path, filename)
        with open(path, 'r') as f:
            d = json.load(f)
            if type(d) != dict or d is None:
                raise ValueError(f"File {path} does not contain a dictionary")
            self.info.append(d)

    def print_epoch_summary(self, epoch: int) -> None:
        print("\nInfo summary:")
        for k, v in self.info[epoch].items():
            print(f"\t{k}: {sum(v) / len(v):.3f}")
        print()