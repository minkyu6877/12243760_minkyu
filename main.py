import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

SCALE_FACTOR = 100.0

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedCascadePINN().to(device)

weight_path = 'model.pt'
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()

def residuals_fn(p, p_bs, d_hat_u):
    distances = np.sqrt((p_bs[0, :] - p[0])**2 + (p_bs[1, :] - p[1])**2)
    return distances - d_hat_u

def get_nls_prior(d_hat_u, p_bs):
    p0 = np.array([np.mean(p_bs[0, :]), np.mean(p_bs[1, :])])
    res = least_squares(residuals_fn, p0, loss='huber', f_scale=1.0, args=(p_bs, d_hat_u))
    return res.x

def advanced_cascade_algorithm(d_hat_u, p_nls, p_bs):
    calc_dist = np.sqrt((p_bs[0, :] - p_nls[0])**2 + (p_bs[1, :] - p_nls[1])**2)
    residuals_val = calc_dist - d_hat_u
    
    X_combined = np.concatenate((d_hat_u, p_nls, residuals_val))
    X_tensor = torch.tensor(X_combined, dtype=torch.float32).unsqueeze(0).to(device) / SCALE_FACTOR
    
    with torch.no_grad():
        p_pred_scaled = model(X_tensor)
    
    p_pred = (p_pred_scaled.squeeze(0).cpu().numpy()) * SCALE_FACTOR
    return p_pred

def main():
    mat_path = 'DH_FR1.mat'
    try:
        data = sio.loadmat(mat_path, squeeze_me=False)
    except FileNotFoundError:
        return np.zeros((2, 1))
    
    if 'p_bs' in data:
        BS_positions = np.asarray(data['p_bs'], dtype=float)
    else:
        BS_positions = np.asarray(data['BS_positions'], dtype=float)
        
    d_hat = np.asarray(data['d_hat'], dtype=float)
    num_user = d_hat.shape[1]
    
    p_hat_step1 = np.zeros((2, num_user))
    p_hat_final = np.zeros((2, num_user))
    
    for u in range(num_user):
        p_nls = get_nls_prior(d_hat[:, u], BS_positions)
        p_hat_step1[:, u] = p_nls
        p_hat_final[:, u] = advanced_cascade_algorithm(d_hat[:, u], p_nls, BS_positions)

    if 'p' in data:
        p_true = np.asarray(data['p'], dtype=float)
        err_step1 = np.mean(np.linalg.norm(p_true - p_hat_step1, axis=0))
        err_final = np.mean(np.linalg.norm(p_true - p_hat_final, axis=0))
        
        print("\n" + "="*55)
        print("📊 [최종 고도화] Advanced Cascade 파이프라인 성능 평가")
        print("="*55)
        print(f"1단계 (Huber NLS 단독)        : {err_step1:.4f} m")
        print(f"2단계 (Advanced PINN 적용 후) : {err_final:.4f} m")
        print("="*55 + "\n")

    return p_hat_final

if __name__ == "__main__":
    _ = main()