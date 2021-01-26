import os

input_dir = "D:\\data\\filer\\"

paths = os.listdir(input_dir)
print(len(paths))
#print(paths[:10])

handles = []
for path in paths:
    parant = False
    if "(" in path and ")" in path:
        parant = True
    if ("@" in path):
        index = path.index("@")+1
    elif ("___" in path):
        index = path.index("___")+3
    else:
        continue
    i = index
    for s in path[index:]:
        if not s.isalnum():
            break
        i += 1
    try:
        index2 = path.index("_-")
        handle = path[index:index2].strip("_")
    except:
        handle = path[index:i].strip("_")
    handles.append(handle)
print(handles)
handles = set(handles)
print(handles)
links = ""
for handle in handles:
    if handle == "":
        continue
    if "___" in handle:
        continue
    links += "https://www.instagram.com/"+handle + "\n"
links = links[:-1]
print(links)
print(len(links))

with open("links3.txt", "w") as f:
    f.write(links)