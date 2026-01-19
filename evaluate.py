import numpy as np
import argparse
from src.evaluation import Evaluator
from tqdm import tqdm

def main(task = "reach_top_left", model= "td2_cfm", epoch = None):
    evaluator = Evaluator(model=model, task=task, epoch=epoch, num_samples=64, disable=True)
    estimates = [evaluator.evaluate() for _ in tqdm(range(10), desc="Evaluations")]
    emd = [res['EMD'] for res in estimates]
    msev = [res['MSE(V)'] for res in estimates]
    nll = [res['NLL'] for res in estimates]
    print("Task:", task)
    print("Model:", model)
    print("EMD:", np.mean(emd), f'({np.std(emd)})')
    print("MSE(V):", np.mean(msev), f'({np.std(msev)})')
    print("NLL:", np.mean(nll), f'({np.std(nll)})')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='reach_top_left')
    parser.add_argument('--model', type=str, default='td2_cfm')
    parser.add_argument('--epoch', type=int, default=None)
    args = parser.parse_args()
    main(task=args.task, model=args.model, epoch=args.epoch)