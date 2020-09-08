import argparse
import numpy as np
import matplotlib.pyplot as plot
import matplotlib


font = {'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)


def histogram_averaging(histograms):

    plot_name = histograms[0].split('_histogram')[0]+'_average_histogram.png'
    print('[INFO] Average histogram will be saved here: {}'.format(plot_name))
    hists = np.asarray([np.loadtxt(x) for x in histograms])
    bin_file_paths = [x.replace('counts', 'bins') for x in histograms]
    try:
        bins = np.asarray([np.loadtxt(x) for x in bin_file_paths])
    except OSError:
        raise Exception('Bin files were not found in the histograms folder. '
                        'Averaging cannot be performed.')
    if not np.isclose(bins, bins[0]).all():
        raise Exception('It seems the bin files provided are not the same.'
                        ' Histograms with different bins cannot be averaged')
    else:
        bin_scheme = bins[0]

    average_hist = np.mean(hists, axis=0)
    
    _, ax = plot.subplots(figsize=(30, 15))
    bar = ax.hist(bin_scheme[:-1], bins=bin_scheme,
                  weights=average_hist)
#     yticks = np.linspace(0, max(bar[0]), 10)
#     ytick_labels = (yticks*seconds_x_frame).astype(int)
#     ax.set_yticks(yticks)
#     ax.set_yticklabels(ytick_labels)
    cm = plot.cm.get_cmap('RdYlBu_r')
    bin_centers = 0.5 * np.asarray(
        (bar[1][:-1] + bar[1][1:]))
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, bar[2]):
        plot.setp(p, 'facecolor', cm(c))
    plot.xlabel("Temperature [C°]")
    plot.ylabel("Average time spent [seconds]")
    plot.savefig(plot_name)
    plot.show()


def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--histograms", nargs='+', type=str,
        help="List of histograms that has to be averaged.")

    args = vars(ap.parse_args())
    
    histogram_averaging(args["histograms"])
    
    print('Done')


if __name__ == "__main__":
    main()
