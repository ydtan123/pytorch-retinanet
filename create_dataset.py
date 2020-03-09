#!/usr/bin/python3
import argparse
import cv2
import os
import pathlib
import random

def getLocation(txtfile, img_file):
    #0 0.056711 0.629032 0.030246 0.204301
    symbols = []
    img = cv2.imread(str(img_file))
    imgh, imgw, _ = img.shape
    with open(str(txtfile)) as f:
        for line in f:
            data = [d for d in line.split(' ')]
            if (len(data) < 5):
                print("Incorrect data {0} in {1}".format(line, txtfile))
                continue
            dval = int(data[0])
            if (dval < 0 or dval >= 20 ):
                print("Invalid number {0} in {1}".format(data[0], txtfile))
                continue
            cx = float(data[1]) * imgw 
            cy = float(data[2]) * imgh
            w = float(data[3]) * imgw / 2
            h = float(data[4]) * imgh / 2
            symbols.append((data[0], int(cx - w), int(cy - h), int(cx + w), int(cy + h)))
    return symbols

def draw_boxes(f, img_file, letters):
    im = cv2.imread(f)
    for l in letters:
        cv2.rectangle(im, (l[1], l[2]), (l[3], l[4]), (255, 0, 0), 1)
    cv2.imwrite(img_file, im)


def write_labels(file, labels):
    if (labels is None or len(labels) == 0):
        return
    with open(file, 'w') as lf:
        for fname, flabels in labels:
            for l in flabels:
                lf.write("{},{},{},{},{},{}\n".format(fname, l[1], l[2], l[3], l[4], l[0]))

if __name__ == '__main__':
    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Paths to the input images separated by ,", default='')
    parser.add_argument("-d", "--debug", action='store_false', help="Debug mode", default=False)
    parser.add_argument("-t", "--tests", type=int, help="the number of test images.", default=100)
    parser.add_argument("-v", "--validation", type=int, help="the number of validation images.", default=500)
    parser.add_argument("-n", "--number", type=int, help="the number of trainging images.", default=1000000)

    parser.add_argument("-a", "--data-root", type=str, help='dataset root')

    args = vars(parser.parse_args())
    if not os.path.exists(args['data_root']):
        os.makedirs(args['data_root'])
    img_with_box = os.path.abspath(os.path.join(args['data_root'], 'JPEGImages_with_boxes'))
    if args["debug"] and not os.path.exists(img_with_box):
        os.makedirs(img_with_box)

    needed = args["tests"] + args["validation"] + args["number"]
    file_dict = {}
    total_count = 0
    repeated = 0
    label_count = 0
    files = []
    image_paths = args["image"].split(',')
    for path in image_paths:
        print("Processing images in {}".format(path))
        n_imgs = 0
        for f in pathlib.Path(path).glob("**/*.jpg"):
            total_count += 1
            fstr = str(f)
            if (fstr in file_dict):
                print("{0} has more than one copy".format(f))
                repeated += 1
                continue
            file_dict[fstr] = 1
            files.append(f.with_suffix(".jpg"))
            n_imgs += 1
            if (total_count > needed):
                break
        print("{} images, total: {}".format(n_imgs, total_count))

    random.shuffle(files)

    test_count = 0
    train_count = 0
    file_info = []
    for f in files:
        fstr = str(f)
        txtfile = f.with_suffix(".txt")
        if (not os.path.isfile(str(txtfile))):
            print("GT file for {0} does not exist".format(f))
            continue

        img_labels = getLocation(txtfile, f)
        if (len(img_labels) == 0):
            print("Did not find labels in {0}".format(f))
            continue
        label_count += 1
        
        if (args["debug"]):
            dst =  os.path.join(img_with_box, f.stem + ".jpg")
            print("draw {} boxes for {}, {}".format(len(img_labels), fstr, dst))
            draw_boxes(fstr, dst, img_labels)

        file_info.append((fstr, img_labels))
        if (needed < len(file_info)):
            break

    write_labels(os.path.join(args["data_root"], "test.csv"), file_info[:args["tests"]])
    write_labels(os.path.join(args["data_root"], "val.csv"),
        file_info[args["tests"]:args["tests"] + args["validation"]])
    write_labels(os.path.join(args["data_root"], "train.csv"),
        file_info[args["tests"] + args["validation"]:])

    print("Orgin Images: {}, Dup: {}, Labeled Images: {}, Train:{}, Test:{}, val:{}"
        .format(total_count, repeated, label_count, label_count - args["tests"] - args["validation"], args["tests"], args["validation"]))
