import urllib.request
import re
fp = urllib.request.urlopen("")
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

#with open("website.txt", "w", encoding="utf8") as f:
#    f.write(mystr)


#with open("website.txt", "r", encoding="utf8") as f:
#    mystr = f.read()
index = mystr.index('<div class="gridWrapper">')
mystr = mystr[index:]
mystr = mystr.replace("\n", "")
mystr = mystr.replace("  ", "")
mystr = mystr.replace("\t", " ")
regexp = r'<li class="listItem[\s\S]+?href="(.*?)"[\s\S]+?<\/li>'
#print([x.group() for x in re.finditer( regexp, mystr)])
links = re.findall(regexp,mystr)
with open("links2.txt", "w") as f:
    for link in links:
        f.write("site" + link + "\n")
#print(mystr)