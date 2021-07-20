def plot_confusion_matrix(cm, classes, save_name, normalize=False, title='Confusion matrix', figsize=(12, 10),
                          cmap=plt.cm.Blues, save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.switch_backend('agg')
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        plt.savefig(save_name)
    
def draw(x, y):
    from matplotlib.colors import ListedColormap
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    print("TSNE: fitting start...")
    tsne = TSNE(2, n_jobs=4, perplexity=100)
    Y = tsne.fit_transform(x)

    # matplotlib_axes_logger.setLevel('ERROR')
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
#     labels = ['wordpress','oracle','svn','apache','excel','matlab','visual-studio','cocoa','osx','bash','unknown']
    id_to_label = {i: label for i, label in enumerate(labels) }
    y_true = pd.Series(y)
    plt.style.use('ggplot')
    n_class = y_true.unique().shape[0]
    colors = ( 'gray','lightgreen', 'plum','DarkMagenta','SkyBlue','PaleTurquoise','DeepPink','Gold','Orange','Brown','DarkKhaki')

    #cmap = plt.cm.get_cmap("tab20", n_class)

    fig, ax = plt.subplots(figsize=(9, 6), )
    la = [i for i in range(n_class)]
    la = sorted(la,reverse=True)
    cmap = ListedColormap(colors)
    for idx, label in enumerate(la):
        ix = y_true[y_true==label].index
        x = Y[:, 0][ix]
        y = Y[:, 1][ix]
        ax.scatter(x, y, c=cmap(idx), label=id_to_label[label], alpha=0.5)
    #     ax.scatter(x, y, c=np.random.rand(3,), label=label, s=100)

    # Shrink current axis by 20%
    ax.set_title('proto_loss')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('output/tsne-CDAC+2.pdf', bbox_inches='tight')
#     plt.show()

def plot_curve(points):
    centers = [[] for x in range(len(points[0]))]
    print('centers',centers)
    for clusters in points:
        clusters = clusters.cpu().detach().numpy()
        for i,c in enumerate(clusters):
            centers[i].append(c)
    print('centers',centers)
    plt.figure()
    markers = ['o', '*', 's', '^', 'x', 'd', 'D', '|', '_', '+', 'h', 'H', '.', ',', 'v', '<', '>', '1', '2', '3', '4', 'p']
    labels = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','unknown']
    
    x = [i for i in range(len(centers[0]))]
    for i,y in enumerate(centers):
        plt.plot(x,y,label=labels[i], marker=markers[i])
        
    plt.xlabel('Epoch')
    plt.ylabel('Decision Boundary $\Delta$')
    plt.legend()
    plt.title('50% Known Classes on StackOverflow')
    plt.show()
    plt.savefig('curve.pdf')
