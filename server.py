import flwr as fl
import torch
import multiprocessing as mp
import argparse
from flower_helpers import set_weights, get_weights, Net, test, FedAvgMp


"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
# import os
# if os.environ.get("https_proxy"):
#     del os.environ["https_proxy"]
# if os.environ.get("http_proxy"):
#     del os.environ["http_proxy"]
"""
Uncomment and comment the one you need depending if you need gpu or not
to test your model fast enough.
"""
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE: str = "cpu"
model = None


def get_eval_fn(model):
    """Get the evaluation function for server side.

    Parameters
    ----------
    modelreturn self.super()
    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights):
        """Evaluation function for server side.

        Parameters
        ----------
        weights
            Updated model weights to evaluate.

        Returns
        -------
        loss
            Loss on the test set.
        accuracy
            Accuracy on the test set.
        """

        # Load model
        # set_weights(model, weights)
        set_weights(model, weights)
        # Prepare multiprocess
        manager = mp.Manager()
        # We receive the results through a shared dictionary
        return_dict = manager.dict()
        # Create the process
        p = mp.Process(target=test, args=(weights, return_dict))
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
        # Del everything related to multiprocessing
        del (manager, return_dict, p)
        return float(loss), float(accuracy)

    return evaluate


# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    parser.add_argument(
        "-mec",
        type=int,
        default=5,
        help="Max number of evaluated clients for the evaluation step.",
    )
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    mec = int(args.mec)
    # Determine the fraction of clients we want to evaluate on
    frac_eval = fc if mec / fc > 1 else mec / fc
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    # Create a new model for testing
    net = Net().to(DEVICE)
    # Define the strategy
    strategy = FedAvgMp(
        fraction_fit=float(fc / ac),
        fraction_eval=frac_eval,
        min_fit_clients=fc,
        min_eval_clients=mec,
        min_available_clients=ac,
        eval_fn=get_eval_fn(net),
        initial_parameters=get_weights(net),
    )
    fl.server.start_server(
        "[::]:8080", config={"num_rounds": rounds}, strategy=strategy
    )
