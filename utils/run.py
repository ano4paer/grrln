import os

if __name__ == "__main__":
    lang = 'java'
    lr = 0.001
    batch = 32
    gru = 128
    dw = 128
    epoch = 5
    # model_type = ['astnn', 'astnn_no_tree', 'astnn_res', 'proto']
    model_type = "proto"
    times = 5
    model_name = str(lang) + "_" + str(model_type) + "_" + str(lr) + "_" + str(batch) + "_" + str(dw) + "_" + str(
        gru) + "_" + str(times)

    for time in range(times):
        cmd = "python train.py" + " --lang " + str(lang) + " --lr " + str(lr) + " --batch " + str(
            batch) + " --gru " + str(gru) + " --dw " + str(dw) + " --epoch " + str(epoch) + " --model_type " + str(
            model_type) + " --times " + str(time)
        print(cmd)
        os.system(cmd)
