import numpy as np

def analyse_repeated_benchmark(train_accs, val_accs, test_accs, avg_runtimes, repeat):
    # Averages
    avg_train_acc = np.mean(train_accs)
    avg_val_acc = np.mean(val_accs)
    avg_test_acc = np.mean(test_accs)
    avg_runtime = np.mean(avg_runtimes)

    # Standard deviation
    sd_train_acc = np.std(train_accs)
    sd_val_acc = np.std(val_accs)
    sd_test_acc = np.std(test_accs)
    sd_runtime = None

    # Standard error of the mean
    sem_train_acc = sd_train_acc / np.sqrt(repeat)
    sem_val_acc = sd_val_acc / np.sqrt(repeat)
    sem_test_acc = sd_test_acc / np.sqrt(repeat)
    sem_runtime_acc = None

    return (avg_train_acc, sd_train_acc, sem_train_acc), (avg_val_acc, sd_val_acc, sem_val_acc), (avg_test_acc, sd_test_acc, sem_test_acc), (avg_runtime, sd_runtime, sem_runtime_acc)
