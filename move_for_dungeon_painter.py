import os
import re

def list_dir_recursive(*dirs,regex = ".*"):
    pattern = re.compile(regex)
    files = []
    layer = []
    for dir in dirs:
        #temp_files = os.listdir(dir)
        files = [dir]
        new_files = [dir]
        index = 0
        while True:
            if os.path.isdir(files[index]):
                new_listdir = os.listdir(files[index])
                files[index:index+1] = [os.path.join(files[index],p) for p in new_listdir]
                tobeadded = [os.path.join(dir,p) for p in new_listdir]
                for i in range(len(tobeadded)):
                    while tobeadded[i] in new_files:
                        tobeadded[i] += "_"
                new_files[index:index+1] = tobeadded
            else:
                index += 1
                if index == len(files):
                    break
    return files,new_files

input_folder="D:\steam\SteamLibrary\steamapps\common\Dungeon Painter Studio\data\collections\custom_structurs\objects"
output_folder=input_folder
#files,new_files = list_dir_recursive(input_folder)
#print(files)
#print(new_files)

for dir in os.listdir(input_folder):
    print(os.path.join(input_folder,dir))
    files, new_files = list_dir_recursive(os.path.join(input_folder,dir))
    for i in range(len(files)):
        try:
            os.rename(files[i],new_files[i])
        except:
            print(files[i])
            print(new_files[i])