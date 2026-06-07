import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
if torch.cuda.is_available():
    print(f"Default device set to: {torch.cuda.get_device_name(torch.cuda.current_device())}")


class DeepONet(nn.Module):
    def __init__(self, branch_input, num_layers, hidden_layers, branch_output, trunk_output):
        super().__init__()

        self.branch_net = nn.ModuleList()
        self.trunk_net = nn.ModuleList()

        self.branch_net.append(nn.Linear(branch_input, hidden_layers[0]))
        self.branch_net.append(nn.GELU())
        
        for i in range(1, num_layers + 1):
            self.branch_net.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.branch_net.append(nn.GELU())

        self.trunk_net = nn.ModuleList()


class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, p):
        super().__init__()

        layers =  []

        dims = [input_dim] + hidden_layers + [p]  # combines [input_dim, hidden1, hidden2, ..., p]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)  # should be size (batch, p)

        return out


class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, p):
        super().__init__()

        layers = []

        dims = [input_dim] + hidden_layers + [p] # combines [input_dim, hidden1, hidden2, ..., p]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)  # should be size (num_sims, p)

        return out


class ScalarMLP(nn.Module):  # To learn scalar outputs, takes in P latent dim output from branch net
    def __init__(self, p, hidden_layers, scalar_out):
        super().__init__()

        layers = []

        dims = [p] + hidden_layers + [scalar_out]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        #print(f"x shape: {x.shape}")
        out = self.net(x)  # should be size (batch, 15)


        return out


# Deep ONet with branch and trunk nets
class Full_DeepONet(nn.Module):
    def __init__(self, p, branch_input, branch_hidden, trunk_input, trunk_hidden, scalar_hidden, scalar_out): 
        '''
        branch_input: Input to branch net corresponding to the 5 input params
        trunk_input: Input to trunk net corresponding to (u, v, num_channels)
        scalar_out: The output from the scalar MLP corresponding to the 15 output scalars
        '''
        super().__init__()

        self.branch_net = BranchNet(branch_input, branch_hidden, p)  # Branch net for input params
        self.trunk_net = TrunkNet(trunk_input, trunk_hidden, p)  # Trunk net for image behavior on coordinates
        self.scalar_head = ScalarMLP(p, scalar_hidden, scalar_out)  # Scalar MLP for learnable scalar outputs
        
        self.bias = nn.Parameter(torch.zeros(1))  # For a learnable bias on the image output 

    def forward(self, x_param, x_imag):
        '''
        x_param: The input parameters--the input to the branch net
        x_imag: The output images--the input to the trunk net
        '''
        phi = self.branch_net(x_param)
        psi = self.trunk_net(x_imag)

        image_pred = phi @ psi.T + self.bias
        scalar_pred = self.scalar_head(phi)
        
        return image_pred, scalar_pred
    

# Coordinate system for the images
def make_coordinates(image_H, image_W, num_channels):
    '''
    Coordinate system is (u,v) for (x,y)
    You're defining it over (image_H = 64, image_W = 64, num_channels = 4) images
    '''

    u = torch.linspace(0, 1, image_H)  #x-coords
    v = torch.linspace(0, 1, image_W)  #y-coords
    U, V = torch.meshgrid(u, v, indexing='ij')  # This is [h, w]

    coords = []
    for i in range(num_channels):
        ch_norm = i / (num_channels - 1)  # Normalized elements from channel
        ch_tensor = torch.full((image_H, image_W), ch_norm)  # Fills [h, w] with ch_norm
        coords.append(torch.stack([U, V, ch_tensor], dim=-1).reshape(-1, 3))

    result = torch.cat(coords, dim=0).to(device)

    return result


def fourier_encode(y, num_freqs=16):
    freqs = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs).to(y.device)
    # y: [n_points, 3]
    y_freq = y.unsqueeze(-1) * freqs * np.pi  # size (n_points, 3, num_freqs)
    y_freq = y_freq.reshape(y.shape[0], -1)   # size (n_points, 3*num_freqs)
    return torch.cat([torch.sin(y_freq), torch.cos(y_freq)], dim=-1)
    # output size (n_points, 6*num_freqs)


def train_DeepONet(params, images, scalars, num_epochs, branch_depth, trunk_depth, scalar_depth, hidden_size, learn_rate, output_freq, p, h, w, c, lambda_s):
    '''
    params: Input params (10000, 5)
    images: Output images (10000, 64, 64, 4)
    scalars: Output scalars (10000, 15)
    (branch, trunk, scalar)_depth: Number of layers in individual branch, trunk, scalar nets 
    hidden_size: Width in each of the branch, trunk, scalar nets
    learn_rate: Learning rate for training
    Output_freq: The divisor of total epochs you want to see print statements for
    p: Number of basis dimensions
    (h, w, c): Image height, width, channels 
    lambda_s: Tunable hyperparameter for the total loss function
    '''

    torch.manual_seed(42); np.random.seed(42)

    num_sims = params.shape[0]  # total number of simulations

    params = params.to(device)
    scalars = scalars.to(device)
    images = images.to(device)

    y_coords = make_coordinates(h, w, c)  # This creates the coordinate system that dictates the behavior of the images on the domain
    y_fourier = fourier_encode(y_coords)  # Fourier encoding of the coordinate system to capture high frequency behavior in the images  
    print(f"y coords: {y_coords.shape}")
    print(f"y fourier:")

    dataset = TensorDataset(params, images, scalars)
    train_size = int(0.8 * num_sims)
    test_size = params.shape[0] - train_size
    train_params, test_params = random_split(dataset, [train_size, test_size])  # Train test split
    train_loader = DataLoader(train_params, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_params, batch_size=128, shuffle=False)

    branch_layers = [hidden_size] * branch_depth
    trunk_layers = [hidden_size] * trunk_depth
    scalar_layers = [hidden_size] * scalar_depth

    model = Full_DeepONet(p, branch_input=params.shape[1], branch_hidden=branch_layers, trunk_input=y_fourier.shape[1], trunk_hidden=trunk_layers,
                          scalar_hidden=scalar_layers, scalar_out=scalars.shape[1])  # The whole Deep ONet model with branch net, trunk net, and scalar MLP
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = nn.MSELoss()

    train_losses = []; test_losses = []
    model.train()
    for i in range(num_epochs):
        train_loss = 0.0
 
        for x_batch, img_batch, sca_batch in train_loader:
            optimizer.zero_grad()
 
            img_pred, sca_pred = model(x_batch, y_fourier)  # Calling model with image x and y coordinates
 
            loss_img    = loss_fn(img_pred, img_batch)  # Individual loss for the learned image from Deep ONet
            loss_scalar = loss_fn(sca_pred, sca_batch)  # Individual loss for the learned scalar from Scalar MLP
            loss        = loss_img + lambda_s * loss_scalar  # Joint loss with both the individual losses
 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if i % output_freq == 0:
            model.eval()
            with torch.no_grad():
                test_loss = sum(
                    loss_fn(model(xb.to(device), y_fourier)[0], ib.to(device)).item()
                    for xb, ib, _ in test_loader
                )
            print(f"Epoch {i:4d} | train loss: {train_loss/len(train_loader):.4f} "
                  f"| test image loss: {test_loss/len(test_loader):.4f}")
        
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
 
    return model, train_losses, test_losses 

def model_evaluate(model, params, images, scalars, h, w, c):
    model.eval()
    y_coords = make_coordinates(h, w, c).to(device)
    y_fourier = fourier_encode(y_coords)
    
    N = params.shape[0]
    params  = params.to(device)
    scalars = scalars.to(device)

    with torch.no_grad():
        img_pred, sca_pred = model(params, y_fourier)

    # --- Scalars ---
    sca_pred_np = sca_pred.cpu().numpy()
    sca_true_np = scalars.cpu().numpy()

    if params.shape[0] > 1:
        scalar_r2 = 1 - np.sum((sca_true_np - sca_pred_np)**2) / np.sum((sca_true_np - sca_true_np.mean(0))**2)
        scalar_rel_err = np.mean(np.abs(sca_true_np - sca_pred_np) / (np.abs(sca_true_np) + 1e-8))

        # --- Images ---
        img_pred_np = img_pred.cpu().numpy().reshape(N, h, w, c)
        img_true_np = images.reshape(N, h, w, c).numpy()
        image_r2 = 1 - np.sum((img_true_np - img_pred_np)**2) / np.sum((img_true_np - img_true_np.mean(0))**2)
        image_rel_err = np.mean(np.abs(img_true_np - img_pred_np) / (np.abs(img_true_np) + 1e-8))

    else:  # For only one sample
        scalar_r2 = 1 - np.sum((sca_true_np - sca_pred_np)**2) / np.sum((sca_true_np)**2)
        scalar_rel_err = np.mean(np.abs(sca_true_np - sca_pred_np) / (np.abs(sca_true_np) + 1e-8))

        # --- Images ---
        img_pred_np = img_pred.cpu().numpy().reshape(N, h, w, c)
        img_true_np = images.reshape(N, h, w, c).numpy()
        image_r2 = 1 - np.sum((img_true_np - img_pred_np)**2) / np.sum((img_true_np)**2)
        image_rel_err = np.mean(np.abs(img_true_np - img_pred_np) / (np.abs(img_true_np) + 1e-8))

    print(f"Scalar R²:         {scalar_r2:.4f}")
    print(f"Scalar rel. error: {scalar_rel_err:.4f}")
    print(f"Image  R²:         {image_r2:.4f}")
    print(f"Image  rel. error: {image_rel_err:.4f}")

    return img_pred_np, sca_pred_np, scalar_r2, image_r2


def invariance_predict(model, params, h, w, c):
    model.eval()
    y_coords = make_coordinates(h, w, c).to(device)
    y_fourier = fourier_encode(y_coords).to(device)

    params = params.to(device)
    num_sims = params.shape[0]

    with torch.no_grad():
        img_pred, sca_pred = model(params, y_fourier)

    img_pred_np = img_pred.cpu().numpy().reshape(num_sims, h, w, c)
    sca_pred_np = sca_pred.cpu().numpy()

    return img_pred_np, sca_pred_np