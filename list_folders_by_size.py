import os

folderfolder="filer"

folders = os.listdir(folderfolder)

foldersnumbers = {}

for folder in folders:
    foldersnumbers[folder] = len(os.listdir(os.path.join(folderfolder,folder)))
print(foldersnumbers)
print(sorted(foldersnumbers,key=foldersnumbers.get,reverse=True))