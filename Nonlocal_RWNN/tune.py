
import torch, os

os.system('pip install ray[tune] optuna')
os.environ['TUNE_WARN_SLOW_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S'] = '30'
torch.cuda.set_per_process_memory_fraction(0.45, 0)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"


from run import run
from ray.air import session

import torch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import Repeater
from ray import tune
import ray

from types import SimpleNamespace
from datetime import datetime, timedelta
import argparse
import shutil
import time

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
print(f"Available GPUs: {num_gpus}")
print(f"Available CPUs: {num_cpus}")
ray.init(num_cpus=num_cpus, num_gpus=num_gpus)


def ts():
    return datetime.now().strftime("%H:%M:%S")


class TrialCounter(tune.Callback):
    def __init__(self, target_num_trials):
        self.started = 0
        self.completed = 0
        self.target = target_num_trials
        self.start_time = time.time()
        self.trial_start_times = {}

    def on_trial_start(self, iteration, trials, trial, **info):
        self.started += 1
        self.trial_start_times[trial.trial_id] = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶ STARTED {self.started}/{self.target}  id={trial.trial_id}",
              flush=True)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        val_acc = trial.last_result.get('val_acc', None)
        test_acc = trial.last_result.get('test_acc', None)

        avg_time_per_trial = elapsed_time / self.completed
        remaining_trials = self.target - self.completed
        eta_seconds = remaining_trials * avg_time_per_trial
        eta_time = datetime.now() + timedelta(seconds=eta_seconds)
        eta_str = eta_time.strftime('%H:%M:%S')

        trials_per_hour = self.completed / (elapsed_time / 3600)
        trial_duration = current_time - self.trial_start_times.get(trial.trial_id, current_time)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✔ COMPLETED {self.completed}/{self.target}  "
              f'id={trial.trial_id}  val_acc={val_acc:.2f}  test_acc={test_acc:.2f}  '
              f'duration={trial_duration:.1f}s  speed={trials_per_hour:.1f}/hr  ETA={eta_str}', flush=True)


def objective(config):
    checkpoint = os.path.join(os.getcwd(), 'model.pt')
    config['checkpoint'] = checkpoint
    args = SimpleNamespace(**config)
    # (val_acc, test_acc), _ = run(args)
    # ray.tune.report(dict(val_acc=val_acc, test_acc=test_acc))
    run(args)


def experiment(args):
    if args.split < 0:
        name = datetime.now().strftime('%m%d%Y%H%M%S') + f'_{args.data}'
    else:
        name = datetime.now().strftime('%m%d%Y%H%M%S') + f'_{args.data}_split{args.split}'
    print(f'Experiment name: {name}')

    # local_storage_path = "/tmp/ray_results"
    # final_storage_path = os.path.join(args.log_dir, name)

    storage_root = os.path.join(os.getcwd(), "logs")
    final_storage_path = os.path.join(storage_root, name)
    print(f"[{ts()}] storage_path={storage_root}", flush=True)

    param_space = {
        'dataset': args.data,
        'data_dir': args.data_dir,
        'split': args.split,
        'seed': -1,

        'nhid': tune.randint(32, 128),
        'natt': tune.randint(16, 80),
        'nhead': tune.randint(2, 8),
        'nlayer': 1,
        # 'num_walks_per_node': 8,  # tune.randint(4, 8),
        "num_samples": tune.choice([2, 3, 4, 5, 6, 7, 8]),
        'length': tune.randint(4, 16),
        'dropout': tune.uniform(0.0, 0.4),
        'rnn_nlayer': 1,  # tune.randint(1, 3),
        'activation': 'SiLU',  # tune.choice(['ReLU', 'ELU', 'SiLU']),
        'self_supervise_weight': tune.loguniform(1e-4, 1.0),
        'consistency_weight': tune.loguniform(1e-4, 1.0),
        'consistency_temperature': tune.uniform(0.0, 1.0),
        'directed': args.directed,
        "walker": args.walker,
        'max_dist': tune.sample_from(lambda spec: spec.config["length"] + 2),

        'optimizer': 'Adam',
        'learning_rate': tune.loguniform(1e-4, 1e-1),
        'weight_decay': tune.loguniform(1e-6, 1e-2),
        'max_epoch': 1000,
        'patience': 100,
        'checkpoint': 0,
        'lr_decay_factor': 0.1,
        'lr_patience': 500000,  # large value to disable lr scheduler

        'in_tune_session': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    }

    # Join the search for α/c/walk_mode only if nonlocal
    if args.walker == 'nonlocal':
        param_space.update({
            'alpha': tune.loguniform(0.01, 150),
            'c': tune.uniform(0.0, 1.0),
            'walk_mode': tune.choice(['exp', 'power']),
        })

    max_concurrent_trials = int(num_gpus // args.gpu_per_trial)
    print(f'Max concurrent trials: {max_concurrent_trials}')

    tune_config = tune.TuneConfig(
        metric='val_acc',
        mode='max',
        search_alg=Repeater(OptunaSearch(), 3),
        num_samples=args.num_trials,
        max_concurrent_trials=max_concurrent_trials
    )

    counter = TrialCounter(target_num_trials=args.num_trials)

    run_config = tune.RunConfig(
        name=name,
        storage_path=storage_root,
        verbose=0,
        callbacks=[counter],
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {
            'cpu': min(1, num_cpus // max_concurrent_trials),
            'gpu': args.gpu_per_trial
        }),
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )

    print(f"Running tuning with local storage: {storage_root}")
    results = tuner.fit()

    best_result = results.get_best_result(metric='val_acc', mode='max')
    print(f'\nBest config: {best_result.config}')
    print(f"Best final val acc: {best_result.metrics['val_acc']:.2f}")
    print(f"Best final test acc: {best_result.metrics['test_acc']:.2f}")

    print(f"Moving results from {storage_root} to {final_storage_path}")

    os.makedirs(args.log_dir, exist_ok=True)
    local_experiment_path = os.path.join(storage_root, name)
    shutil.move(local_experiment_path, final_storage_path)
    print(f"Results moved to: {final_storage_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Cora')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--directed', type=int, default=0)
    parser.add_argument('--split', type=int, default=-1)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('--gpu_per_trial', type=float, default=0.5, help='Number of GPUs per trial')
    parser.add_argument('--use_local_storage', action='store_true',
                        help='Use local storage for results, then move to final storage path, for K8s compatibility')
    parser.add_argument('--walker', type=str, default='nonlocal', choices=['uniform', 'nonlocal'],
                        help='uniform=RUM baseline; nonlocal=Pα walker')

    args = parser.parse_args()
    experiment(args)
