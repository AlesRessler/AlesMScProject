import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from models.neuralnetwork.spherical_cnn import SCNNModel
from preprocessing.data_augmentation import extend_dataset_with_origin_reflections
from preprocessing.data_transformations import convert_coords_from_cartesian_to_spherical
from sphericalharmonics.spherical_fourier_transform import get_spherical_fourier_transform
from sphericalharmonics.spherical_fourier_transform import get_design_matrix
from sphericalharmonics.spherical_fourier_transform import get_inverse_spherical_fourier_transform

number_of_shells = 1
output_size = 45
learning_rate = 0.001
final_learning_rate = 0.0001
batch_size = 1000
num_batches = 10000
num_batches_lr_reduction = 10000
max_degree = 8

all_dwis = np.load('./data/planar/no_rotation/diffusion_weighted_signals.npy')
all_fODFs = np.load('./data/planar/no_rotation/fODF_sh_coefficients.npy')
all_qhat = np.load('./data/planar/no_rotation/gradient_orientations.npy')
all_bvals = np.load('./data/planar/no_rotation/b_values.npy')
all_dwis_sh_coefficients = np.load("./data/planar/no_rotation/diffusion_weighted_signals_sh_coefficients.npy")

thetas, phis = convert_coords_from_cartesian_to_spherical(all_qhat[0])

design_matrix = get_design_matrix(max_degree = max_degree ,number_of_samples=len(all_bvals[0]), thetas=thetas, phis=phis)
spherical_fourier_transform = get_spherical_fourier_transform(design_matrix)

inverse_spherical_fourier_transform = get_inverse_spherical_fourier_transform(design_matrix)

spherical_fourier_transform = torch.from_numpy(spherical_fourier_transform).float()
inverse_spherical_fourier_transform = torch.from_numpy(inverse_spherical_fourier_transform).float()

all_dwis_sh_coefficients = np.expand_dims(all_dwis_sh_coefficients, 1)

# Create the neural network
model = SCNNModel(number_of_shells, output_size, max_degree, spherical_fourier_transform,
                  inverse_spherical_fourier_transform)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data arrays
train_data = all_dwis_sh_coefficients[:60000]
train_targets = all_fODFs[:60000]
val_data = all_dwis_sh_coefficients[60000:80000]
val_targets = all_fODFs[60000:80000]
test_data = all_dwis_sh_coefficients[80000:]
test_targets = all_fODFs[80000:]

# Lists to store errors for plotting
train_errors = []
val_errors = []
test_errors = []

device = torch.device('mps')
model.to(device)

# Training loop
for batch in range(num_batches):

    if (batch % 1000 == 0):
        print(batch)

    # Adjust learning rate
    if batch == num_batches - num_batches_lr_reduction:
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_learning_rate

    indices = torch.randperm(len(train_data))[:batch_size]

    inputs = torch.FloatTensor(train_data[indices])
    targets = torch.FloatTensor(train_targets[indices])

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_errors.append(loss.item())

    # Calculate validation and test errors
    if batch % 100 == 0:
        with torch.no_grad():
            val_inputs = torch.FloatTensor(val_data).to(device)
            val_target = torch.FloatTensor(val_targets).to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_target)
            val_errors.append(val_loss.item())

            test_inputs = torch.FloatTensor(test_data).to(device)
            test_target = torch.FloatTensor(test_targets).to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_target)
            test_errors.append(test_loss.item())

# Plot errors
plt.plot(train_errors, label='Train Error')
plt.plot(val_errors, label='Validation Error')
plt.plot(test_errors, label='Test Error')
plt.xlabel('Batches')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()