import os
import torch
import yaml
import argparse
import numpy as np
from core.dataset import MMDataEvaluationLoader
from models.TFMamba import build_model
from core.metric import MetricsTop

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='configs/eval_mosei.yaml')
opt = parser.parse_args()

def main():
    with open(opt.config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    args['base']['missing_rate_eval_test'] = 0.8
    eval_loader = MMDataEvaluationLoader(args)

    model = build_model(args).to(device)
    dataset_name = args['dataset']['datasetName']
    ckpt_path = os.path.join('ckpt', dataset_name, f'best_MAE_1111.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
    model.eval()


    gamma_values = [0, 5, 10, 15, 18, 20, 25, 30]
    results_mae = []
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(dataset_name)

    print("\n" + "="*60)
    print("开始执行 LTI 稳态阻尼 (Gamma) 敏感性压测...")
    print("="*60)

    for g in gamma_values:
        model.set_dtf_gamma(g)
        
        y_pred, y_true = [], []
        for data in eval_loader:
            complete_input = (None, None, None)
            incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
            label = data['labels']['M'].to(device)
            
            with torch.no_grad():
                out = model(complete_input, incomplete_input)
                y_pred.append(out['sentiment_preds'].cpu())
                y_true.append(label.cpu())
                
        pred_tensor = torch.cat(y_pred, dim=0)
        truth_tensor = torch.cat(y_true, dim=0)
        
        test_results = metrics(pred_tensor, truth_tensor)
        cur_mae = test_results['MAE']
        results_mae.append(cur_mae)
        
        print(f"Gamma (\u03b3) = {g:2d}  |  MAE = {cur_mae:.4f}")

    os.makedirs('log/results', exist_ok=True)
    np.save(f'log/results/{dataset_name}_gamma_sensitivity_MAE.npy', np.array(results_mae))
    np.save(f'log/results/{dataset_name}_gamma_values.npy', np.array(gamma_values))
    print("\n[导师系统]: 测试完成！")

if __name__ == '__main__':
    main()