# Scaling Flower with Multiprocessing

## A short Introduction of Federated Learning: 

As recent technologies produce more and more data as they evolve, accessing large quantity of them and training accurate models on them is becoming more and more accessible. However it raises concerns privacy concerns and, to ensure their privacy, people are currently protected by a number of laws depending on where they live (for instance GDPR in Europe). The traditional approach of machine learning consisting in accumulating data in a single spot to train models can’t be blindly applied to personnal data. 

Then Google released in 2016 a new paradigm for training models in such context called Federated Learning and applied it to its Google Keyboard app \need sauce\. It was introduced to leverage the problem in domain difference between the publicly available datasets they trained their model on and the private data users would produce. 
As specified in the Federated Learning book \need sauce\ , in order for this paradigm to work, it needs to respect 4 main principles which are: 

    • At least 2 entities want to train a model and have data they own and are ready to use it.
    • During training, the data doesn’t leave its original owner.
    • The model can be transferred from one entity to another through protected means.
    • The resulting model performances are a good approximation of the ideal model trained with all data owned by a single entity.

Now this tells us multiple things about Federated Learning, people involved in the training of the model must have given their consent to this training, instead of the data transiting between entities it’s now the model, the transaction has to be protected (using cryptographic means for instance or other techniques such as Differential Privacy) and finally according means must be set to get a good approximation of ideal model. The last point is also telling us that Federated Learning can’t always be applied neither. Its biggest drawback is that, as it is, Federated Learning is sensible to attack from the inside \need sauce\, is not guaranteed to converge \need sauce\ and needs enough clients to achieve its results \need sauce\. However, when applied correctly, it can produces models that couldn’t be obtained through normal means like Google and its Google Keyboard.

As of now, only a few frameworks exist to implement it, since it’s a fairly new concept. Tensorflow has developped its own version called Tensorflow Federated \need sauce\ but Pytorch as yet to see its own implementation. They do exist frameworks compatible with PyTorch such as PySyft, developped by OpenMined, another alternative is Flower \need sauce\ which will be the main focus of this post.

## The GPU problem:

As I said before, it’s really easy to scale to as much clients as your CPU allows you to. For simple models, CPU is enough and there is no need to extend training on GPU. However when using bigger models or bigger datasets, you might want to move to GPU in order to greatly improve training speed. This is where you can encounter a problem in scaling your Federated setting. Indeed when accessing the GPU, CUDA will automatically allocate a fixed amount of memory so that it has enough room to work with before asking for more. However, this memory can’t be freed at all, until the process exits. This means that if you are launching a 100 clients and sample 10 of them, everytime a client will try to use the GPU, there will be leftover memory that can’t be released in the GPU everytime a new client is sampled. It means that in the long term, your GPU needs as much memory as if you’d sample all 100 clients at once. 

“ Insert GPU images here”


## How to solve the issue: 

This problem that  you might have encountered, can be solved quite easily actually. Since the memory is not released until the process accessing it is released, then we simply need to encapsulate the part of our code that need to access the GPU in a subprocess, waiting for it to be terminated until we can continue to execute our program. Multiprocessing is the solution, and I will show you how to do it using PyTorch and Flower.

## Why use Flower:

Flower is a recent framework for Federated Learning, created in 2020. Contrary to Tensorflow Federated and Pysyft which are linked to a single framework, Flower can be used with all of them by design. It focuses on giving the tool for applying Federated Learning efficiently and allows you to focus on the training itself. Implementing a Federated version with Flower is really simple (only 20 lines of codes is enough) and the rewriting needed to adapt a centralized code to a federated one is really low. As well, the range of compatible devices (from mobile devices, to Raspberry Pi, servers and others) is quite large. It’s architecture also allows scalability to 1000s of clients as shown in their paper \need sauce\. It is overall a really great framework to experiment with. 

## How it’s done:

You will first need to install the required dependencies for your project. You should create a new virtual environment to do so. I personally recommend to use poetry `need link here`, as it is easy to install and solves the dependencies for you, but you could also use conda or even a simple environment with venv. I provided all the code used in this blug in the github repository here: `nom du repo` and you can use the pyproject.toml file and the poetry.lock file to initialize your new environment with “poetry install”. If you want to start from scratch, just use “poetry init” to initialize your virtual environment, then use “poetry add library-name” to add the desired library (you will at least need PyTorch, Torchivision, Flower and Numpy to run the example). 

Since this example is based on the “Quickstart Pytorch tutorial” `need sauce` from the Flower documentation, I highly recommend to check it out before continuing, since it shows the basics.

### Helper file 

First we’re gonna buil ourselves a flower_helpers.py file where we’ll put some functions and classes that will come handy later. Starting with the imports, we have:

```python
from collections import OrderedDict
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
```

Basic imports torch imports for working on CIFAR10 and a flower strategy import since we will need to slightly change the FedAvg strategy for our use case. We then define the device on which we want to compute the training and test steps:

```python
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```


Next, we need to define how we’re going to load the data:

```python
def load_data(train=True):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10("./dataset", train=train, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader
```


A simple CNN model from “PyTorch: A 60 Minute Blitz”:

```python
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

So far, nothing changed from the original flower tutorial, now things are going to become different. Since we can’t keep in the client’s memory the model, we need to define a way to get our model’s weights so that the client can keep track of them, for that we move the `get_parameters` and `set_parameters` function for the flower tutorial and put them in our helper file: 

```python
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
```

```python
def set_weights(model, weights) -> None:
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
```


Now we can define the training function that will be called every time a client wants to train its model:

```python
def train(epochs, parameters, return_dict):
    """Train the network on the training set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    trainloader = load_data(train=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    # Prepare return values
    return_dict["parameters"] = get_weights(net)
    return_dict["data_size"] = len(trainloader)
```


This function takes three parameters, the number of local epochs we want to make, the new parameters of the global model and a return dictionary that will act as our return values to give back to the client the updated model, the size of the local dataset and other metrics we’d like to give back like the loss or accuracy. Same for the testing function:

```python
def test(parameters, return_dict):
    """Validate the network on the entire test set."""
    # Create model
    net = Net().to(DEVICE)
    # Load weights
    if parameters is not None:
        set_weights(net, parameters)
    # Load data (CIFAR-10)
    testloader = load_data(train=False)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    # Prepare return values
    return_dict["loss"] = loss
    return_dict["accuracy"] = accuracy
    return_dict["data_size"] = len(testloader)
```


Finally, we need to define our custom strategy. These are not mentioned in the QuickStart tutorial but strategies are the classes that determine how the server will aggregate the new weights, how it will evaluate clients, etc. The most basic strategy is the FedAvg (for federated average) which we will use to implement our own. Flower already gives you a way for precising the number of you want to use to evaluate your model through the initial parameters of the FedAvg strategy, however this is only true for the evaluations carried between each round. Indeed, after the final round, the flower server will perform a last evaluation step with all clients available to verify the model’s performances. That wouldn’t be a problem in a real case scenario but this could actually backfire in ours, we want to especially avoid a scenario that could involve an overflow of GPU memory requirements. This is why we will be performing evaluation in this tutorial only on the server side and we will remove this functionality. This is done through the `configure_evaluate` method of the strategy, which we need to override:

```python
class FedAvgMp(FedAvg):
    """This class implements the FedAvg strategy for Multiprocessing context."""

    def configure_evaluate(self, rnd, parameters, client_manager):
        """Configure the next round of evaluation. Returns None since evaluation is made server side.
        You could comment this method if you want to keep the same behaviour as FedAvg."""
        return None
```

### Client file

We're done with the helper file, we can now switch to the client side. Starting with imports: 

```python
import flwr as fl
import multiprocessing as mp
from flower_helpers import train, test
```

The next step is to implement our own Client class so it can connect to a flower server. We need to derive from the NumpyClient flower class. For that we need to implement 4 methods, namely `get_parameters`, `set_parameters`, `fit` and `evaluate`. We will also add an attribute called `parameters`, where we will keep tracks of the model's weights:

```python
class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.parameters = None
```

`get_parameters` and `set_parameters` are straightforward, they are just getter and setter:

```python
def get_parameters(self):
    return self.parameters

def set_parameters(self, parameters):
    self.parameters = parameters
```

Then the `fit` method is where we train the model, it receives two parameters: the new parameters from the global model and a config dictionnary containing the configuration for the current round. It's inside `fit` that we're going to launch a subprocess so we can use GPU withtout being worried about lingering memory.

```python
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
```
As you can see, the function returns the newly updated parameters, the size of the local dataset and a dictionnary (here empty) that could contain different metrics. Last we have the `evaluate` method, similar to `fit` but used for evaluation. In our case we could chose to simply implement the minimum required as we won't evaluate our clients. But I will give here the full implementation:

```python
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

```

We only have to wrap all of this in `main`, set the `spawning` way to create new subprocesses (not the default with Python under 3.8) and start our client on local port 8080: 

```python
def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")

    # Flower client
    class CifarClient(fl.client.NumPyClient): # .....

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())


if __name__ == "__main__":
    main()
```

### Server file

With the client being done we can now work our way toward the server! In the original tutorial, starting the server take as much as a single line of code! But here we will perform server side evaluation and using a custom strategy so things are slightly different. Beginning with imports:

```python
import flwr as fl
import multiprocessing as mp
import argparse
from flower_helpers import Net, FedAvgMp, get_weights, test
```

First, we need to define the way we are going to evaluate our model on the server side. We need to encapsulate the function in `get_eval_fn` which tells the server how to get the function. The evaluation is almost identical to the one I gave for the client side and you could actually merge part of it.

```python
def get_eval_fn():
    """Get the evaluation function for server side.

    Returns
    -------
    evaluate
        The evaluation function
    """

    def evaluate(weights):
        """Evaluation function for server side."""
        
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
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate
```
Then we can start the `__main__` and load arguments and set the spawn method:

```python
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
    args = parser.parse_args()
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
```

We then get a fresh network so we can initialize the weights for the federated loop:

```python
 # Create a new fresh model to initialize parameters
net = Net()
init_weights = get_weights(net)
# Convert the weights (np.ndarray) to parameters
init_param = fl.common.weights_to_parameters(init_weights)
# del the net as we don't need it anymore
del net
```

Finaly, define the strategy and launch the server on port 8080.

```python
# Define the strategy
strategy = FedAvgMp(
    fraction_fit=float(fc / ac),
    min_fit_clients=fc,
    min_available_clients=ac,
    eval_fn=get_eval_fn(),
    initial_parameters=init_param,
)
fl.server.start_server(
    "[::]:8080", config={"num_rounds": rounds}, strategy=strategy
)
```

### Bash file

The only thing left to do is to launch our server and clients! We write a nice bash file so we only have to run it to start experimenting:

```bash
#!/bin/bash

# Loading script arguments
NBCLIENTS="${1:-2}" # Nb of clients launched by the script (default to 2)
NBMINCLIENTS="${2:-2}" # Nb min of clients before launching round (default to 2)
NBFITCLIENTS="${3:-2}" # Nb of clients sampled for the round (default to 2)
NBROUNDS="${4:-3}" # Nb of rounds (default to 3)

python server.py -r $NBROUNDS -fc $NBFITCLIENTS -ac $NBMINCLIENTS &
sleep 5 # Sleep for N seconds to give the server enough time to start, increase if clients can't connect
for ((nb=0; nb<$NBCLIENTS; nb++))
do
    python client.py &
done


# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# If still not stopping you can use `killall python` or `killall python3` or ultimately `pkill python`
sleep 86400
```

### Running
Now, by simply running `./run.sh` in your terminal, you should see the following output: 

```
INFO flower 2021-05-20 11:42:40,062 | app.py:73 | Flower server running (insecure, 3 rounds)
INFO flower 2021-05-20 11:42:40,062 | server.py:118 | Getting initial parameters
INFO flower 2021-05-20 11:42:40,062 | server.py:300 | Received initial parameters from strategy
INFO flower 2021-05-20 11:42:40,062 | server.py:120 | Evaluating initial parameters
Files already downloaded and verified
INFO flower 2021-05-20 11:42:43,255 | server.py:123 | initial parameters (loss, other metrics): 721.5202655792236, {'accuracy': 0.1044}
INFO flower 2021-05-20 11:42:43,255 | server.py:133 | FL starting
DEBUG flower 2021-05-20 11:42:45,084 | connection.py:36 | ChannelConnectivity.IDLE
DEBUG flower 2021-05-20 11:42:45,084 | connection.py:36 | ChannelConnectivity.IDLE
DEBUG flower 2021-05-20 11:42:45,084 | connection.py:36 | ChannelConnectivity.CONNECTING
DEBUG flower 2021-05-20 11:42:45,084 | connection.py:36 | ChannelConnectivity.READY
INFO flower 2021-05-20 11:42:45,084 | app.py:61 | Opened (insecure) gRPC connection
DEBUG flower 2021-05-20 11:42:45,085 | connection.py:36 | ChannelConnectivity.READY
INFO flower 2021-05-20 11:42:45,085 | app.py:61 | Opened (insecure) gRPC connection
DEBUG flower 2021-05-20 11:42:45,085 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
Files already downloaded and verified
Files already downloaded and verified
DEBUG flower 2021-05-20 11:42:59,792 | server.py:260 | fit_round received 2 results and 0 failures
Files already downloaded and verified
INFO flower 2021-05-20 11:43:03,011 | server.py:148 | fit progress: (1, 657.4171632528305, {'accuracy': 0.2509}, 19.755342989999917)
INFO flower 2021-05-20 11:43:03,011 | server.py:199 | evaluate_round: no clients selected, cancel
DEBUG flower 2021-05-20 11:43:03,011 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
Files already downloaded and verified
Files already downloaded and verified
DEBUG flower 2021-05-20 11:43:17,812 | server.py:260 | fit_round received 2 results and 0 failures
Files already downloaded and verified
INFO flower 2021-05-20 11:43:21,045 | server.py:148 | fit progress: (2, 518.9745894670486, {'accuracy': 0.3839}, 37.790048795000985)
INFO flower 2021-05-20 11:43:21,045 | server.py:199 | evaluate_round: no clients selected, cancel
DEBUG flower 2021-05-20 11:43:21,046 | server.py:251 | fit_round: strategy sampled 2 clients (out of 2)
Files already downloaded and verified
Files already downloaded and verified
DEBUG flower 2021-05-20 11:43:35,516 | server.py:260 | fit_round received 2 results and 0 failures
Files already downloaded and verified
INFO flower 2021-05-20 11:43:38,736 | server.py:148 | fit progress: (3, 481.05929803848267, {'accuracy': 0.4381}, 55.48068630399939)
INFO flower 2021-05-20 11:43:38,736 | server.py:199 | evaluate_round: no clients selected, cancel
INFO flower 2021-05-20 11:43:38,736 | server.py:172 | FL finished in 55.48090230299931
INFO flower 2021-05-20 11:43:38,736 | app.py:109 | app_fit: losses_distributed []
INFO flower 2021-05-20 11:43:38,736 | app.py:110 | app_fit: metrics_distributed {}
INFO flower 2021-05-20 11:43:38,736 | app.py:111 | app_fit: losses_centralized [(0, 721.5202655792236), (1, 657.4171632528305), (2, 518.9745894670486), (3, 481.05929803848267)]
INFO flower 2021-05-20 11:43:38,736 | app.py:112 | app_fit: metrics_centralized {'accuracy': [(0, 0.1044), (1, 0.2509), (2, 0.3839), (3, 0.4381)]}
INFO flower 2021-05-20 11:43:38,736 | server.py:199 | evaluate_round: no clients selected, cancel
INFO flower 2021-05-20 11:43:38,737 | app.py:129 | app_evaluate: no evaluation result
DEBUG flower 2021-05-20 11:43:38,739 | connection.py:68 | Insecure gRPC channel closed
DEBUG flower 2021-05-20 11:43:38,739 | connection.py:68 | Insecure gRPC channel closed
INFO flower 2021-05-20 11:43:38,739 | app.py:72 | Disconnect and shut down
INFO flower 2021-05-20 11:43:38,739 | app.py:72 | Disconnect and shut down

```

If for some reason, you get an error telling you the clients can't connect, make sure the server has enough time to set up before clients try to connect to it. Another reason might be because of a known bug with GRPC and python, you can try adding the following lines in your server and client files:

```python
import os
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]
```

All the code is available on GitHub `sauce`. You can now launch as many clients as your CPU allows you and managing your GPU memory as it fits your needs. This concludes this tutorial Hope it will come handy for you, don't hesitate to leave a feedback!