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

    return (
        (avg_train_acc, sd_train_acc, sem_train_acc),
        (avg_val_acc, sd_val_acc, sem_val_acc),
        (avg_test_acc, sd_test_acc, sem_test_acc),
        (avg_runtime, sd_runtime, sem_runtime_acc),
    )


def stringify_statistics(
    experiment_name, train_res_1, val_res_1, test_res_1, runtime_res_1
):
    train_str = f"Train acc {experiment_name}: avg {train_res_1[0]} %, std: {train_res_1[1]} %, sem: {train_res_1[2]} %"
    val_str = f"Val acc {experiment_name}: avg {val_res_1[0]} %, std: {val_res_1[1]} %, sem: {val_res_1[2]}%"
    test_str = f"Test acc {experiment_name}: avg {test_res_1[0]} %, std: {test_res_1[1]} %, sem: {test_res_1[2]} %"
    runtime_str = f"Runtime {experiment_name}: avg {runtime_res_1[0]} s, std: {runtime_res_1[1]} s, sem: {runtime_res_1[2]} s"
    return f"{train_str}\n{val_str}\n{test_str}\n{runtime_str}\n"
