import numpy as np
import spectral as spy
from spectral import spy_colors
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 前设置
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
def save_legend(save_path, n_classes, class_cmap, include_background=True):
    """
    Save a standalone legend figure.
 
    Args:
        save_path: where to save legend image
        n_classes: int(gt.max())
        class_cmap: your ListedColormap
        include_background: whether to show class 0
    """
 
    patches = []
 
    if include_background:
        patches.append(Patch(color=class_cmap(0), label='0'))
 
    for i in range(1, n_classes + 1):
        patches.append(Patch(color=class_cmap(i), label=str(i)))
 
    # Create figure (vertical legend)
    fig, ax = plt.subplots(figsize=(2, n_classes * 0.4))
    ax.axis('off')
 
    legend = ax.legend(
        handles=patches,
        loc='center',
        frameon=False,
        ncol=1,
        fontsize=10,
        handlelength=4,
        handleheight=4
    )
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)
 
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
 
 

def visualize_cluster_centers(centers, epoch=0, n_classes=9, sub_clusters_per_class=3, method='pca'):
    """
    Visualize cluster centers (27x128) in 2D using t-SNE or PCA.

    Args:
        centers: np.ndarray of shape (27, 128)
        n_classes: number of parent classes (default 9)
        sub_clusters_per_class: number of subclusters per class (default 3)
        method: 'tsne' or 'pca'
    """
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=5, learning_rate=100)
    if method == 'pca':
        reducer = PCA(n_components=2)
    centers_2d = reducer.fit_transform(centers)

    # Assign parent class & subcluster indices
    parent_classes = np.arange(len(centers)) // sub_clusters_per_class
    sub_ids = np.arange(len(centers)) % sub_clusters_per_class

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        centers_2d[:, 0], centers_2d[:, 1],
        c=parent_classes, cmap='tab10', s=80, edgecolor='k'
    )

    # Annotate each sub-cluster center
    for i, (x, y) in enumerate(centers_2d):
        plt.text(x + 0.3, y, f"{parent_classes[i]}-{sub_ids[i]}", fontsize=9)

    plt.title(f"Cluster Centers ({len(centers)} total) visualized by {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Parent class ID")
    plt.tight_layout()
    plt.savefig(f"visual_predict_{method}_{epoch}.png")


def visualize_predict(gt,predict_label,save_predict_path,save_gt_path,only_vis_label=False):
    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label,(row,col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt==0,gt,predict)
    else:
        vis_predict = predict

    n_classes = gt.max()
    colors = plt.cm.jet(np.linspace(0, 1, n_classes))
    colors = np.vstack([[[0,0,0,1]], colors])  # Black for background
    class_cmap = ListedColormap(colors)

    height, width = vis_predict.shape
    fig, ax = plt.subplots()
    ax.imshow(vis_predict, cmap=class_cmap, vmin=0, vmax=n_classes)
    plt.axis('off')
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)
    

    # plt.imshow(vis_predict, cmap=class_cmap, vmin=0, vmax=n_classes)
    # plt.savefig(save_predict_path)
    # plt.imshow(gt, cmap=class_cmap, vmin=0, vmax=n_classes)
    plt.savefig(save_predict_path)
    plt.close(fig)
    # spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    # spy.save_rgb(save_gt_path, gt, colors=spy_colors)
    
    
    
    fig, ax = plt.subplots()
    ax.imshow(gt, cmap=class_cmap, vmin=0, vmax=n_classes)
    plt.axis('off')
    fig.set_size_inches(width / 100.0, height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.savefig(save_gt_path)
    plt.close(fig)
    
    save_legend(
        save_path="legend.png",
        n_classes=int(gt.max()),
        class_cmap=class_cmap,
        include_background=False   # or False if you don’t want 0
    )
 