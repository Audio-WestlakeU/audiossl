from matplotlib import pyplot as plt
def plot_spec(x,save_path):
    t = range(0,x.shape[0])
    f = range(0,x.shape[1])
    plt.pcolormesh(x)
    plt.xlabel('frequency',fontsize=20)
    plt.ylabel('time',fontsize=20)
    plt.savefig(save_path)
    plt.close()




