import os

# RVI solution
# for index in range(3, 10):
#    run_depth_pct = 'python main.py --policy=RVI --videoId=' + str(index)
#    os.system(run_depth_pct)

# SAC solution
for index in range(1, 20):
    run_depth_pct = 'python main.py --policy=RVI --saliTestNum=' + str(index)
    os.system(run_depth_pct)