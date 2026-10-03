import csv
import torch
from tabulate import tabulate
from collections import defaultdict

def count_parameters(
        model: torch.nn.Module,
        grouped: bool = False,
        save_to_file: bool = False,
        file_path: str = "params.csv",
        only_trainable: bool = False,
        min_params: int = 0,
        print_results: bool = True
) -> tuple[int, int]:
    """
    Analyzes and summarizes the parameters of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to analyze.
    grouped : bool, optional
        If True, groups parameters by top-level module name. If False, lists all parameters individually.
    save_to_file : bool, optional
        If True, saves the parameter table to a file.
    file_path : str, optional
        Path to save the parameter table. Should end with '.csv' or '.txt'.
    only_trainable : bool, optional
        If True, includes only parameters that require gradients (trainable parameters).
    min_params : int, optional
        Minimum number of parameters required for a parameter (or group) to be included in the output.

    Returns
    -------
    total_params : int
        Total number of parameters (after filtering, if applied).
    trainable_params : int
        Total number of trainable parameters (after filtering, if applied).

    Notes
    -----
    - Outputs a formatted table of parameters to the console.
    - The table is sorted by descending number of parameters.
    - Can optionally save the output table to a CSV or TXT file.
    """

    if grouped:
        grouped_params = defaultdict(lambda: {"total": 0, "trainable": 0})

        for name, param in model.named_parameters():
            if only_trainable and not param.requires_grad:
                continue

            base_module = name.split('.')[0] if '.' in name else name
            count = param.numel()

            if count < min_params:
                continue

            grouped_params[base_module]["total"] += count
            if param.requires_grad:
                grouped_params[base_module]["trainable"] += count

        sorted_items = sorted(grouped_params.items(), key=lambda x: x[1]["total"], reverse=True)

        table = []
        total_all = trainable_all = 0
        for module_name, stats in sorted_items:
            total = stats["total"]
            trainable = stats["trainable"]
            non_trainable = total - trainable
            table.append([module_name, total, trainable, non_trainable])
            total_all += total
            trainable_all += trainable

        headers = ["Module", "Total Params", "Trainable", "Non-trainable"]

    else:
        table = []
        total_all = trainable_all = 0
        for name, param in model.named_parameters():
            count = param.numel()
            if only_trainable and not param.requires_grad:
                continue
            if count < min_params:
                continue
            trainable = param.requires_grad
            table.append([name, count, trainable])
            total_all += count
            if trainable:
                trainable_all += count

        headers = ["Parameter", "Param Count", "Trainable"]
        table.sort(key=lambda x: x[1], reverse=True)

    if save_to_file:
        ext = file_path.split('.')[-1].lower()
        with open(file_path, mode='w', newline='') as f:
            if ext == 'csv':
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in table:
                    writer.writerow(row)
            elif ext == 'txt':
                f.write(tabulate(table, headers=headers, tablefmt="pretty"))
            else:
                print(f"[!] Unsupported file extension: .{ext} — use .csv or .txt")

        print(f"\n[✓] Parameter summary saved to '{file_path}'")
    if print_results:
        print(tabulate(table, headers=headers, tablefmt="pretty"))
        print("\nSummary:")
        print(f"Total parameters: {total_all:,}")
        print(f"Trainable parameters: {trainable_all:,}")
        print(f"Non-trainable parameters: {total_all - trainable_all:,}")

    return total_all, trainable_all


