import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

EPOCHS = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
LAMBDA_PHYSICS = 0.5
SCALE_FACTOR = 100.0

def residuals(p, p_bs, d_hat_u):
    distances = np.sqrt((p_bs[0, :] - p[0])**2 + (p_bs[1, :] - p[1])**2)
    return distances - d_hat_u

def get_nls_prior(d_hat_u, p_bs):
    p0 = np.array([np.mean(p_bs[0, :]), np.mean(p_bs[1, :])])
    res = least_squares(residuals, p0, loss='huber', f_scale=1.0, args=(p_bs, d_hat_u))
    return res.x

class AdvancedCascadePINN(nn.Module):
    def __init__(self):
        super(AdvancedCascadePINN, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(38, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_block = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        self.relu = nn.ReLU()
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.input_layer(x)
        identity = out
        out = self.res_block(out)
        out = self.relu(out + identity)
        return self.output_layer(out)

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
    
    num_users = d_hat_raw.shape[1]
    
    p_nls_raw = np.zeros((2, num_users))
    residuals_raw = np.zeros((18, num_users))
    
    for u in range(num_users):
        p_nls = get_nls_prior(d_hat_raw[:, u], p_bs_raw)
        p_nls_raw[:, u] = p_nls
        
        calc_dist = np.sqrt((p_bs_raw[0, :] - p_nls[0])**2 + (p_bs_raw[1, :] - p_nls[1])**2)
        residuals_raw[:, u] = calc_dist - d_hat_raw[:, u]
    
    X_combined_raw = np.vstack((d_hat_raw, p_nls_raw, residuals_raw))
    
    X = torch.tensor(X_combined_raw.T, dtype=torch.float32) / SCALE_FACTOR
    Y = torch.tensor(p_raw.T, dtype=torch.float32) / SCALE_FACTOR
    BS_pos = torch.tensor(p_bs_raw, dtype=torch.float32) / SCALE_FACTOR
    d_hat_tensor = torch.tensor(d_hat_raw.T, dtype=torch.float32) / SCALE_FACTOR
    
    dataset = torch.utils.data.TensorDataset(X, Y, d_hat_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AdvancedCascadePINN().to(device)
    BS_pos = BS_pos.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
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
            
        scheduler.step()
        
    save_path = 'model.pt'
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()