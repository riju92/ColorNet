with open("loss_log.txt") as f:
    for line in f:
        if "Train" in line:
            print(line)
