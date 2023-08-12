import os
import shutil

def create_dir(newdir, empty = True):
    """
    create new folder if the target folder doesnt exist
    """
    CHECK_FOLDER = os.path.isdir(newdir)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(newdir)
        print("created folder : ", newdir)

    else:
        if empty == True:
            ## whether to remove all contents in the current augmented data folder and generate new ones
            shutil.rmtree(newdir)
            print("current augmented data removed")
            os.makedirs(newdir)
            
        print(newdir, "folder already exists.")
    
def best_model_finder(mname):
    folderpath = "models/%s/"%mname
    max_performance = 0
    best_model = ""
    for f in os.listdir(folderpath):
        if "history" in f: continue
        perf = float(f.split("-")[2][:4])
        if perf > max_performance:
            max_performance = perf
            best_model = f
    print("The current best performing model: %s is loaded"%best_model)
    return(folderpath+best_model)
