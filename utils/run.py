import os

if __name__ == "__main__":
    lang = 'c'
    lr = 0.01
    batch = 128
    gru = 64    # fixed value
    dw = 128   # fixed value
    epoch = 5
    times = 5
    model_name = str(lang) + "_" + str(lr) + "_" + str(batch) + "_" + str(dw) + "_" + str(
        gru) + "_" + str(times)

    for time in range(times):
        cmd = "python train.py" + " --lang " + str(lang) + " --lr " + str(lr) + " --batch " + str(
            batch) + " --gru " + str(gru) + " --dw " + str(dw) + " --epoch " + str(epoch) + " --times " + str(time)
        print(cmd)
        os.system(cmd)
