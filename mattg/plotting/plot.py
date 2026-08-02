import matplotlib.pyplot as plt


# TODO: Generalise
def plot(dataset):
    n = 24
    images = []
    labels = []
    for i in range(n):
        img, label = dataset[i]
        images.append(img.squeeze(0))  # (1,28,28) -> (28,28)
        labels.append(label)

    fig, axes = plt.subplots(1, n, figsize=(12, 2))
    axes = axes.ravel()
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img, cmap="gray")
        ax.set_title(str(label))
        ax.axis("off")

    fig.tight_layout()
    plt.show()