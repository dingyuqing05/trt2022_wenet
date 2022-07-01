import os 
#import random 

# 2022-06-16 00:09:57,278 INFO Epoch 25 CV info cv_loss 62.99785318527701



def read_file(filepath):
    fd = open(filepath, 'r')
    res = fd.readlines()
    fd.close()
    return res

def write_file(filepath, data):
    fd = open(filepath, 'w')
    cc = len(data)
    for i in range(cc):
        fd.write(data[i])
    fd.close()

def create_data(src_dir, dst_dir, cur_dir):
    sdir = src_dir + "/" + cur_dir + "/"
    ddir = dst_dir + "/" + cur_dir + "/"
    os.mkdir(ddir)
    
    data_percent = 0.1
    stext = read_file(sdir + 'text')
    total = int( len(stext) * data_percent )
    stext = stext[:total]
    write_file(ddir + 'text', stext)

    stext = read_file(sdir + 'wav.scp')
    total = int( len(stext) * data_percent )
    stext = stext[:total]
    write_file(ddir + 'wav.scp', stext)

    

def main():
    src_dir = "./data_bk"
    dst_dir = "./data"

    if os.path.exists(dst_dir):
        os.system("rm -rf " + dst_dir)
        #os.rmdir(dst_dir)

    os.mkdir(dst_dir)

    create_data(src_dir, dst_dir, "dev")
    create_data(src_dir, dst_dir, "test")
    create_data(src_dir, dst_dir, "train")


main()

