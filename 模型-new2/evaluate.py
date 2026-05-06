import numpy as np
import torch
from models.causal_cnn_gru import CausalCNNGRU
from models.baselines import GRUBaseline, LSTMBaseline, MLPBaseline
from prepare_data import causal_moving_average

def compute_metrics(u, e_ref_future, threshold=10):
    mae = np.mean(np.abs(u - e_ref_future))
    jitter = np.mean(np.abs(u[1:] - u[:-1]))
    spike_rate = np.mean(np.abs(u[1:] - u[:-1]) > threshold)
    sign_flip_x = np.mean((u[1:, 0] * u[:-1, 0]) < 0)
    sign_flip_y = np.mean((u[1:, 1] * u[:-1, 1]) < 0)
    sign_flip = (sign_flip_x + sign_flip_y) / 2
    return mae, jitter, spike_rate, sign_flip

def predict_with_model(model, X_test, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_tensor).cpu().numpy()
    return predictions

def evaluate_methods(data, models, device):
    e_raw = data['e_raw']
    e_ref = causal_moving_average(e_raw, window_size=2)
    e_ref_future = np.zeros_like(e_ref)
    e_ref_future[:-1] = e_ref[1:]
    e_ref_future[-1] = e_ref[-1]

    X_test = data['X_test']
    n_total = len(e_raw) - 8
    start_idx = int(n_total * 0.85) + 8
    e_ref_test = e_ref[start_idx:start_idx+len(X_test)]
    e_ref_future_test = e_ref_future[start_idx:start_idx+len(X_test)]
    e_raw_test = e_raw[start_idx:start_idx+len(X_test)]

    results = {}

    # Raw
    results['Raw'] = compute_metrics(e_raw_test, e_ref_future_test)

    # Ref Only
    results['Ref Only'] = compute_metrics(e_ref_test, e_ref_future_test)

    # Prediction Only
    delta_pred = predict_with_model(models['main'], X_test, device)
    u_pred = e_ref_test + delta_pred
    results['Prediction Only'] = compute_metrics(u_pred, e_ref_future_test)

    # Fixed Fusion
    alpha, delta_clip = 0.4, 10
    delta_clipped = np.clip(delta_pred, -delta_clip, delta_clip)
    u_fusion = e_ref_test + alpha * delta_clipped
    results['Fixed Fusion'] = compute_metrics(u_fusion, e_ref_future_test)

    # Adaptive Fusion
    velocity = np.abs(e_ref_test[1:] - e_ref_test[:-1]).sum(axis=1)
    alpha_adaptive = np.clip(0.2 + velocity / 20, 0.2, 0.6)
    u_adaptive = e_ref_test[:-1] + alpha_adaptive[:, None] * delta_clipped[:-1]
    results['Adaptive Fusion'] = compute_metrics(u_adaptive, e_ref_future_test[:-1])

    return results

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data and models...")
    data = np.load('checkpoints/data.npz')
    
    models = {
        'main': CausalCNNGRU().to(device),
        'gru': GRUBaseline().to(device),
        'lstm': LSTMBaseline().to(device),
        'mlp': MLPBaseline().to(device)
    }
    
    models['main'].load_state_dict(torch.load('checkpoints/causal_cnn_gru.pth'))
    models['gru'].load_state_dict(torch.load('checkpoints/gru_baseline.pth'))
    models['lstm'].load_state_dict(torch.load('checkpoints/lstm_baseline.pth'))
    models['mlp'].load_state_dict(torch.load('checkpoints/mlp_baseline.pth'))
    
    print("\n=== Table 2: Main Method Comparison ===")
    results = evaluate_methods(data, models, device)
    for method, (mae, jitter, spike, flip) in results.items():
        print(f"{method} & {mae:.2f} & {jitter:.2f} & {spike:.3f} & {flip:.3f} \\\\")
    
    print("\n=== Table 5: Model Architecture Comparison ===")
    e_ref = causal_moving_average(data['e_raw'], window_size=2)
    n_total = len(data['e_raw']) - 8
    start_idx = int(n_total * 0.85) + 8
    e_ref_test = e_ref[start_idx:start_idx+len(data['X_test'])]
    e_ref_future = np.zeros_like(e_ref_test)
    e_ref_future[:-1] = e_ref_test[1:]
    e_ref_future[-1] = e_ref_test[-1]

    for name, model in [('MLP', models['mlp']), ('GRU', models['gru']),
                        ('LSTM', models['lstm']), ('Causal CNN+GRU', models['main'])]:
        delta_pred = predict_with_model(model, data['X_test'], device)
        u = e_ref_test + 0.4 * np.clip(delta_pred, -10, 10)
        mae, jitter, _, _ = compute_metrics(u, e_ref_future)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name} & {mae:.2f} & {jitter:.2f} & {params} & <1 \\\\")
    
    print("\nEvaluation complete!")
