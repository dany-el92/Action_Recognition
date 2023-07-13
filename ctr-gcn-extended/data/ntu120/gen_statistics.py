import os
import os.path as osp
import re

def main():
    directory='./runs'
    files = os.listdir(directory)
    #exclude files
    files = [f for f in files if osp.isdir(osp.join(directory, f))]
    
    names = []
    action = []
    replication = []
    performer = []
    camera = []
    setup = []

    for filename in files:
        #remove _rgb from filename
        filename = filename[:-4]
        names.append(filename)

        #get number after A and parse to int
        action.append(int(re.search('A(\d+)', filename).group(1)))

        #get number between R and A and parse to int
        replication.append(int(re.search('R(\d+)', filename).group(1)))

        #get number between P and R and parse to int
        performer.append(int(re.search('P(\d+)', filename).group(1)))   

        #get number between C and P and parse to int
        camera.append(int(re.search('C(\d+)', filename).group(1)))

        #get number between S and C and parse to int
        setup.append(int(re.search('S(\d+)', filename).group(1)))

    #create statistics folder
    if not os.path.exists('./statistics'):
        os.makedirs('./statistics')


    #delete ./statistics/skes_available_name.txt if exists
    if os.path.exists('./statistics/skes_available_name.txt'):
        os.remove('./statistics/skes_available_name.txt')
    #print names in statistics/skes_available_name.txt
    with open('./statistics/skes_available_name.txt', 'w') as f:
        for item in names:
            f.write("%s\n" % item)

    #delete ./statistics/label.txt if exists
    if os.path.exists('./statistics/label.txt'):
        os.remove('./statistics/label.txt')
    #print action in statistics/label.txt
    with open('./statistics/label.txt', 'w') as f:
        for item in action:
            f.write("%s\n" % item)

    #delete ./statistics/replication.txt if exists
    if os.path.exists('./statistics/replication.txt'):
        os.remove('./statistics/replication.txt')
    #print replication in statistics/replication.txt
    with open('./statistics/replication.txt', 'w') as f:
        for item in replication:
            f.write("%s\n" % item)

    #delete ./statistics/performer.txt if exists
    if os.path.exists('./statistics/performer.txt'):
        os.remove('./statistics/performer.txt')
    #print performer in statistics/performer.txt
    with open('./statistics/performer.txt', 'w') as f:
        for item in performer:
            f.write("%s\n" % item)

    #delete ./statistics/camera.txt if exists
    if os.path.exists('./statistics/camera.txt'):
        os.remove('./statistics/camera.txt')
    #print camera in statistics/camera.txt
    with open('./statistics/camera.txt', 'w') as f:
        for item in camera:
            f.write("%s\n" % item)

    #delete ./statistics/setup.txt if exists
    if os.path.exists('./statistics/setup.txt'):
        os.remove('./statistics/setup.txt')
    #print setup in statistics/setup.txt
    with open('./statistics/setup.txt', 'w') as f:
        for item in setup:
            f.write("%s\n" % item)

    print("ok")

if __name__ == '__main__':
    main()