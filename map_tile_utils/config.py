
# EDGE DETECTOR:

PLOT_MIN_AREA = 2000     # minimum area of a polygon to identified as plot
PLOT_MAX_AREA = 80000     # maximum area of a polygon to identified as plot
CLUSTER_THRESHOLD = 0.20     # % overlap of larger cluster with small ones required to be removed
NOISE_THRESHOLD = 0.80     # % overlap of small plot with large ones required to be removed
MAX_ASPECT_RATIO = 7.0     # maximum aspect ratio of plot
MIN_COMPACTNESS = 0.1     # minimum compactness of plot
MIN_VERTICES = 3     # minimum number to which vertices of a plot can be reduced to
EROSION_SIZE = (5, 5)     # Internal buffer to skip black borders
GRAY_LIMIT = 25     # Max saturation for a "Gray" road
BLUE_LIMIT = 50     # Max saturation for a "Blue" road
BRIGHTNESS_LIMIT = 150     # Min brightness (Value) for Blue roads
BLUE_HUE_RANGE = (100, 130)     # The "Periwinkle" slice of the HSV wheel



