def make_result_metrics(logs, n):
    """
    logs: {
        key: Tensor (vector)
        ...
    }
    """
    keys = sorted(list(logs.keys()))
    msg = "Test result metric:"
    msg += "\n    idx " + " ".join([f"{key:>10}" for key in keys])
    for i in range(n):
        msg += f"\n    {i:>03d}"
        for j in range(len(keys)):
            val = f"{logs[keys[j]][i].item():.4f}"
            msg += f" {val:>10}"
    msg += "\n----------------------------------------------------------------------------------"
    msg += "\n        " + " ".join([f"{key:>10}" for key in keys])
    msg += "\n    avg"
    for j in range(len(keys)):
        val = f"{logs[keys[j]].mean().item():.4f}"
        msg += f" {val:>10}"

    return msg
    """ <example>
    idx         ch    f1_0001
    000     0.1000     0.1239
    001     0.8000     0.1398
    002     0.0298     0.9486
    -----------------------------------------
    avg     0.0191     0.3958
    """
