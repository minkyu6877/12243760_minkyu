import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio

EPOCHS = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
LAMBDA_PHYSICS = 0.5
SCALE_FACTOR = 100.0

class PINN_Locator(nn.Module):
    def __init__(self):
        super(PINN_Locator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def physics_loss_fn(p_pred, d_hat, bs_positions):
    batch_size = p_pred.shape[0]
    num_bs = bs_positions.shape[1]
    
    bs_pos_batch = bs_positions.unsqueeze(0).repeat(batch_size, 1, 1)
    p_pred_batch = p_pred.unsqueeze(2).repeat(1, 1, num_bs)
    
    calculated_distances = torch.sqrt(torch.sum((p_pred_batch - bs_pos_batch)**2, dim=1))
    
    huber_loss = nn.HuberLoss(delta=1.0)
    loss_phys = huber_loss(calculated_distances, d_hat)
    
    return loss_phys

def main():
    mat_path = 'DH_FR1.mat'
    try:
        data = sio.loadmat(mat_path, squeeze_me=False)
    except FileNotFoundError:
        return

    if 'p_bs' in data:
        p_bs_raw = np.asarray(data['p_bs'], dtype=float)
    else:
        p_bs_raw = np.asarray(data['BS_positions'], dtype=float)
        
    d_hat_raw = np.asarray(data['d_hat'], dtype=float)
    p_raw = np.asarray(data['p'], dtype=float)
    
    X = torch.tensor(d_hat_raw.T, dtype=torch.float32) / SCALE_FACTOR
    Y = torch.tensor(p_raw.T, dtype=torch.float32) / SCALE_FACTOR
    BS_pos = torch.tensor(p_bs_raw, dtype=torch.float32) / SCALE_FACTOR
    d_hat_tensor = torch.tensor(d_hat_raw.T, dtype=torch.float32) / SCALE_FACTOR
    
    dataset = torch.utils.data.TensorDataset(X, Y, d_hat_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PINN_Locator().to(device)
    BS_pos = BS_pos.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    mse_loss_fn = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_Y, batch_d in dataloader:
            batch_X, batch_Y, batch_d = batch_X.to(device), batch_Y.to(device), batch_d.to(device)
            
            optimizer.zero_grad()
            
            p_pred = model(batch_X)
            loss_data = mse_loss_fn(p_pred, batch_Y)
            loss_physics = physics_loss_fn(p_pred, batch_d, BS_pos)
            loss = loss_data + (LAMBDA_PHYSICS * loss_physics)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
    save_path = 'model.pt'
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
