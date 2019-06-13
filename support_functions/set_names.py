import os
import sys



def main():
    i = 1
    src_dir = sys.argv[1]

    name = sys.argv[2]

    for file_name in os.listdir(src_dir):
            new_name = f"{name}_000{i}.jpg"
            if(i > 9):
                    new_name = f"{name}_00{i}.jpg"
            os.rename(src_dir + "\\" + file_name,  src_dir+"\\"+new_name)
            i += 1

    print("Done renaming")

main()
