import h5py
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

class CoalitionNN(nn.Module):
    """
    Neural network to approximate the value of the expected collaborative gain based on a VRP instance and a possible coalition of vehicles.
    """

    def __init__(self):
        super().__init__()

        self.DmNet = nn.Sequential(
            nn.Linear(48, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.coalNET = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, xDm, xC):

        xDmNet = self.DmNet(xDm)
        xCNet = self.coalNET(xC)
        x = torch.cat([xDmNet, xCNet], dim = 1)
        return self.head(x)


if __name__ == '__main__':

    with h5py.File("instances.h5", "r") as f:
        print(list(f.keys()))

        Dm = f['Dm'][:, :, 1:]
        A = f['assign'][:]
        C = f['coalitions'][:]
        v = f['v'][:]

        # Get tensor for input of NN
        Dm = torch.from_numpy(Dm)
        A = torch.from_numpy(A)
        C = torch.from_numpy(C)
        v = torch.from_numpy(v)

    """
    N = 9 (number of customers)
    D = 3 (number of depots)
    N + D = 12 (number of nodes; sum of customers and depots)
    K = 8 = 2 ** D (number of coalitions)   
    print(Dm.shape)
    > (10000, 12, 4)
    print(A.shape)
    > (10000, 12, 8)
    print(C.shape)
    > (10000, 8, 3)
    print(v.shape)
    > (10000, 8)
    """

    # Train a model to predict the expected collaboration gain per coalition so need one instance per coalition and take one specific coalition as input to get its own collaboration as output.
    xDm = Dm.unsqueeze(1).repeat(1, 8, 1, 1)
    xDm = xDm.reshape(-1, 12 * 4)
    """
    [
    [x y d v],
    [x y d v],
    ...
    ]
    
    becomes 
    [ x y d v x y d v ...]
    """
    xC = C.reshape(-1, 3)
    y = v.reshape(-1, 1)

    """
    print(xDm.shape)
    > (80000, 48)
    print(xC.shape)
    > (80000, 3)
    print(y.shape)
    > (80000, 1)
    """

    # Neural Network
    sample_size = xDm.size(0)
    train_size = int(0.8 * sample_size)
    shuffled = torch.randperm(sample_size)
    train_id = shuffled[:train_size]
    test_id = shuffled[train_size:]

    # Create the training set
    xDm_train = xDm[train_id]
    xC_train = xC[train_id]
    y_train = y[train_id]

    # Create the validation set
    xDm_test = xDm[test_id]
    xC_test = xC[test_id]
    y_test = y[test_id]

    # Tools
    model = CoalitionNN()
    writer = SummaryWriter()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = 1e-4)
    epochs = 0
    # TODO: Need to initialize the NN model

    n_dummy = 1
    dm_dummy = torch.ones((n_dummy, 48))
    c_dummy = torch.ones((n_dummy, 3))
    y_dummy  = torch.randn((n_dummy, 1))
    output_dummy = model(dm_dummy, c_dummy)
    print('dummy output shape: ', output_dummy.shape, output_dummy)

    loss = criterion(output_dummy, y_dummy); print('loss: ', loss.item())
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name, param.grad)

    for epoch in range(epochs):

        train_loss = test_loss = 0

        learn = model(xDm_train, xC_train)
        loss = criterion(learn, y_train)
        writer.add_scalar("Loss/train", loss, epoch)

        with torch.no_grad():
            valid = model(xDm_test, xC_test)
            perf = criterion(valid, y_test)
            writer.add_scalar("Loss/test", perf, epoch)
            assert perf.requires_grad == False

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        test_loss += perf.item()

        if epoch % (epochs // 10) == 0:
            # torch.save(model.state_dict(), "coalitions.pth")
            print(f"Epoch {epoch}: Loss (train: {train_loss:.4f}, test: {test_loss:.4f})")
            writer.flush()
    writer.close()

'''
motivation for CoalitionNN:
> Whilst this is not the exact task agents must perform in the collaborative routing scenario, the intuition is that the neural network should still learn useful patterns which are transferable to the full collaborative routing problem.

> we want to know when we add this NN to gym agent whether NN or agent is learning/not learning (to 
'''

# TODO: use and look at the Gym NN agent (after chap 7 or 8) {gdrl}
# TODO: install texify, R plugin