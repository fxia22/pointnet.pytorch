import os
import glob
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
from plyfile import PlyData
from tqdm import tqdm, trange
import time
import pdb
import gc




def preprocess_off(in_file):

    with open(in_file,"rt") as f:
        # some OFF files in original dataset had OFF345 345 344 where 
        # OFF collided with the number. Needs \n
        lines = f.readlines()
    if lines[0] != 'OFF\n':
        with open(in_file,"wt") as f:
            lines[0] = lines[0][0:3] + '\n' + lines[0][3:]
            lines = "".join(lines)
            f.write(lines)
            f.close()
    else:
        f.close()

def offFormat_to_plyFormat(C_file):

    with open(C_file,"rt") as Cf:
        lines = Cf.readlines()
    with open(C_file,"wt") as Cf:
        num_points = lines[1].split()[0]
        num_faces = lines[1].split()[1]
        lines[0] = 'ply\n'
        lines[1] = 'format ascii 1.0\n'+\
                    'element vertex %s'%num_points+'\n'+\
                    'property float x\n'+\
                    'property float y\n'+\
                    'property float z\n'+\
                    'element face %s'%num_faces+'\n'+\
                    'property list uchar int vertex_index\n'+\
                    'end_header\n'
        lines = "".join(lines)
        Cf.write(lines)
        Cf.close()


def full_off_to_ply(p, l, chunksize, num_cpu=None):

    print('searching path', p, 'to convert .off files to .ply')

    if (num_cpu != None):
        start = time.time()
        with Pool(processes = num_cpu) as po0:
            for i in po0.imap_unordered(preprocess_off, l, chunksize=chunksize):
                pass
        with Pool(processes = num_cpu) as po1:
            for i in po1.imap_unordered(offFormat_to_plyFormat, l, chunksize=chunksize):
                pass
        end = time.time()
        print("Muti core full_off_to_ply use %3f sec!" %(end - start))

    else:
        start = time.time()
        for f in l:
            preprocess_off(f)
        for f in l:
            offFormat_to_plyFormat(f)
        end = time.time()
        print("Single core full_off_to_ply use %3f sec!" %(end - start))




def process_txt():
    # Convert test.txt, train.txt, trainval.txt, val.txt data .off to .ply.
    txtNames = glob.glob(r"*.txt")

    for txtName in txtNames:
        with open(txtName,"rt") as file:
            x = file.read()
        with open(txtName, "wt") as file:
            x = x.replace(".off",".ply")
            file.write(x)

def suffix(l):
    # Convert all .off suffix to .ply.
    for i in l:
        tmp = i.with_suffix('.ply')
        os.rename(i, tmp)


def sub_check(f):

    plydata = PlyData.read(f)
    del plydata
    gc.collect()
    

def check(p, chunksize, num_cpu=None):
    # Use PlyData.read(f) check .ply, if ply file broken, sub_check will crush.
    ply = list(p.glob('**/*.ply'))
    progress = tqdm(total=len(ply))

    if (num_cpu != None):
        start = time.time()
        with Pool(processes = num_cpu) as po:
            for i in po.imap_unordered(sub_check, ply, chunksize=chunksize):
                progress.update(1)
        end = time.time()
        print("Muti core check use %3f sec!" %(end - start))
    else:
        start = time.time()
        for f in ply:
            sub_check(f)
            progress.update(1)
        end = time.time()
        print("Single core check use %3f sec!" %(end - start))



if __name__ == '__main__':

    num_cpu = multiprocessing.cpu_count()
    chunksize = 100
    currPATH = os.getcwd().replace('\\','/')
    p = Path(currPATH)
    l = list(p.glob('**/*.off'))

    if isinstance(num_cpu, int):
        print("Now use %d Threads!" %num_cpu)
    else:
        num_cpu = None
        print("Now use single Thread!")

    full_off_to_ply(p, l, chunksize, num_cpu)
    process_txt()
    suffix(l)
    check(p, chunksize, num_cpu)

    print('ur great!')
    print('All .off format has converted!')
