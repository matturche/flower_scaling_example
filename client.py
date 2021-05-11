import flwr as fl
import torch
import multiprocessing as mp
from flower_helpers import train, test

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
# import os
# if os.environ.get("https_proxy"):
#     del os.environ["https_proxy"]
# if os.environ.get("http_proxy"):
#     del os.environ["http_proxy"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self):
            self.parameters = None

        def get_parameters(self):
            return self.parameters

        def set_parameters(self, parameters):
            self.parameters = parameters

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            # Prepare multiprocess
            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            # Create the process
            p = mp.Process(target=train, args=(1, parameters, return_dict))
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Coudln't close the training process: {e}")
            # Get the return values
            new_parameters = return_dict["parameters"]
            data_size = return_dict["data_size"]
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
            return new_parameters, data_size, {}

        def evaluate(self, parameters, config):

            self.set_parameters(parameters)
            # Prepare multiprocess
            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            # Create the process
            p = mp.Process(target=test, args=(parameters, return_dict))
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Coudln't close the evaluating process: {e}")
            # Get the return values
            loss = return_dict["loss"]
            accuracy = return_dict["accuracy"]
            data_size = return_dict["data_size"]
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
            return float(loss), data_size, {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())


if __name__ == "__main__":
    main()
