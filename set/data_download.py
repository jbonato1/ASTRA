import os 
import glob

cwd = os.getcwd()
parent = os.path.dirname(cwd)
print("Set will be saved in this directory: ", parent)

for i in [1,2,3]:
    try:
        os.mkdir(parent+'/set'+str(i))
    except:
        print("Folder {} already exists".format('set'+str(i)))


    #### da inserire i comandi per scaricare i files
