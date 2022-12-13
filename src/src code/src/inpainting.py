import numpy as np
import cvxpy as cp
from multiprocessing import Pool
from imageio import imread
import matplotlib.pyplot as plt

np.random.seed(21)


def killed_pixels(shape, frac_kill):
    """Corrupt pixels of the image"""
    npixels = np.prod(shape)
    num_kill = int(frac_kill * npixels)
    inds = np.random.choice(npixels, num_kill, replace=False)
    return inds

def load_text():
    """Read in overlay text"""
    fname = '../images/misc/text3_1024.png' ## this line should be changed for each specific text image files
    text = np.mean(imread(fname), axis=-1)
    return text


def load_original_image():
    """Read in original image"""
    fname = '../images/original/cat_1024by1024.png' ## this line should be changed for each specific original input image
    output = np.mean(imread(fname), axis=-1)
    return output


def get_noisy_data():
    """Create a noisy distortion of the original image based on the specified percentage"""
    original = load_original_image()
    shape = original.shape
    total = range(np.prod(shape))
    unknown = killed_pixels(shape, frac_kill=0.30) ## this line should be changed based based on the percentage of pixel values to be masked
    known = list(set(total) - set(unknown))
    corrupted = np.zeros(shape)
    rows, cols = np.unravel_index(known, shape)
    corrupted[rows, cols] = original[rows, cols]
    return original, corrupted


def get_text_data():
    """Overprint the text over the original image"""
    original = load_original_image()
    text = load_text()
    corrupted = np.minimum(original + text, 255)
    return original, corrupted


def get_regular_noisy_data():
    """Regular noisy distortion of the original image"""
    original = load_original_image()
    corrupted = original.copy()
    for i in [3, 4, 5, 7, 11]:
        corrupted[0::i, 0::i] = 0
    return original, corrupted


def total_variation(arr):
    """Compute the l1 norm of the discrete gradient"""
    dx = cp.vec(arr[1:, :-1] - arr[:-1, :-1])
    dy = cp.vec(arr[:-1, 1:] - arr[:-1, :-1])
    D = cp.vstack((dx, dy))
    norm = cp.norm(D, p=1, axis=0)
    return cp.sum(norm)


def inpaint(corrupted, rows, cols, verbose=False):
    """CVX minimizes the l1 norm based on the specified constraint"""
    x = cp.Variable(corrupted.shape)
    objective = cp.Minimize(total_variation(x))
    knowledge = x[rows, cols] == corrupted[rows, cols]
    constraints = [0 <= x, x <= 255, knowledge]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK, verbose=verbose)
    return x.value


def compare(corrupted, recovered, original, fname):
    """Generate and print the corrupted and recovered images"""
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5.8, 5.8))
    diff = np.abs(recovered - original)
    images = [corrupted, recovered, diff]
    titles = ['Corrupted', 'Recovered', 'Difference']
    for (ax, image, title) in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.set_axis_off()
    
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight', pad_inches = 0)


def task(data_fun):
    """Call the main functions to execute and fetch the returned outputs"""
    original, corrupted = data_fun()
    rows, cols = np.where(original == corrupted)
    recovered = inpaint(corrupted, rows, cols, verbose=True)
    return corrupted, recovered, original


def main():
    """Execute the functions to recover the corrupted image from text and noise corruption"""
    modes = ['text', 'noisy', 'regular']
    data_funs = [get_text_data, get_noisy_data, get_regular_noisy_data]
    with Pool(len(data_funs)) as pool:
        results = pool.map(task, data_funs)
    for arrays, mode in zip(results, modes): 
        print(f'Saving {mode} image...')
        fname = f'../images/text_results/cat_1024{mode}_results.png' ## file name should be changed for each specific output expected
        compare(*arrays, fname)


if __name__ == '__main__':
    main()
