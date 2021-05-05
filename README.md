# Flower scaling on GPU Example using PyTorch

This is an introductory example to Flower scaling on GPU using PyTorch, it is an adaptation of the "Flower Example using PyTorch" example and provides some insights on how to scale Flower to hundreds of clients using GPU. The problem with GPU and Flower scalability is that every time a new client access to the GPU, memory is automatically allowed to it. CUDA uses a default amount for this memory and sometimes it can be much more than you actually need to run your model. Additionnaly, this memory is not freed until the python process linked to it (our client) is terminated. This means that the memory requirements for adding new clients scale linearly independently of the actual memory required to train a client, making it difficult to attain the hundreds of clients as it is. However, a simple workaround is to move the model and data into the training / testing step and launch it in a spawned subprocess. Combined with the sampling of a fraction of clients we can effectively control the memory usage of the GPU during a federated round. As the example it is adapted from, deep knowledge of PyToch is not necessarily required to run the example and running it is quite easy.

## Project Setup

Start by cloning the example project. 

This will create a new directory called `flower_scaling_example` containing the following files:

```shell
-- pyproject.toml
-- client.py
-- server.py
-- run.sh
-- README.md
```

Project dependencies (such as `torch` and `flwr`) are defined in `pyproject.toml`. We recommend [Poetry](https://python-poetry.org/docs/) to install those dependencies and manage your virtual environment ([Poetry installation](https://python-poetry.org/docs/#installation)), but feel free to use a different way of installing dependencies and managing virtual environments if you have other preferences.

```shell
poetry install
poetry shell
```

Poetry will install all your dependencies in a newly created virtual environment. To verify that everything works correctly you can run the following command:

```shell
python3 -c "import flwr"
```

If you don't see any errors you're good to go!

# Run Federated Learning with PyTorch and Flower

Afterwards you are ready to start the Flower server as well as the clients. You can simply run them in a terminal as follows:

```shell
./run.sh
```

This will create a Flower server and two clients, training for 3 rounds. You can specify addtionnal parameters as follows:

```shell
./run.sh NBCLIENTS NBMINCLIENTS NBFITCLIENTS NBROUNDS
```

`NBCLIENTS` specifies the number of clients you want to launch at once, `NBMINCLIENTS` the minimum number of clients needed to launch a round, `NBFITCLIENTS` the number of clients sampled in a round and `NBROUNDS` the number of rounds you want to train for.


You will see that PyTorch is starting a federated training. If you have issues with clients not connecting, you can try uncommenting these lines in both `server.py` and `client.py`:

```python
import os
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]
```
