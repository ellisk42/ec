# =========== [NEWER VERSION, NOT USING MATPLOTLIB]
import math
import numpy as np
import matplotlib
from scipy.ndimage import gaussian_filter as gf
from skimage import color
from scipy.ndimage import gaussian_filter as gf
import cairo
from math import tan
from math import pi

if False:
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    matplotlib.use('TkAgg')

XYLIM = 3. # i.e., -3 to 3.

SCALES = [0.5, 1., 1.25, 1.5, 2., 2.5, 3]
NPOLY = range(3,7) # range of regular polyogns allowed.
DISTS = [-2., -1.5, -1., -0.5, -0.25, 0, 0.25, 0.5, 1., 1.5, 2.] + [(1/2)*(1)*(1/tan(pi/n)) for n in range(3, 7)] # for making regular polygons
THETAS = [j*(2*pi/8) for j in range(8)] + [-2*pi/6] + [-2*pi/12]



# ============= TRANSFORMATIONS
def _makeAffine(s=1., theta=0., x=0., y=0., order="trs"):
    
    if s is None:
        s=1.
    if theta is None:
        theta=0.
    if x is None:
        x=0.
    if y is None:
        y=0
    if order is None:
        order="trs"
        
    def R(theta):
        T = np.array([[math.cos(theta), -math.sin(theta), 0.], [math.sin(theta), math.cos(theta), 0.], [0.,0.,1.]])
        return T

    def S(s):
        T = np.array([[s, 0., 0.], [0., s, 0.], [0., 0., 1.]])
        return T
    
    def T(x,y):
        T = np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])
        return T
        
    if order == "trs":
        return T(x,y)@(R(theta)@S(s))
    elif order == "tsr":
        return T(x,y)@(S(s)@R(theta))
    elif order == "rts":
        return R(theta)@(T(x,y)@S(s))
    elif order == "rst":
        return R(theta)@(S(s)@T(x,y))
    elif order == "srt":
        return S(s)@(R(theta)@T(x,y))
    elif order == "str":
        return S(s)@(T(x,y)@R(theta))

def _tform(p, T):
    # given prim and affine matrix (T), otuput new p.
    p3 = []
    for i, pp in enumerate(p): # append column of ones to each matrix.
        pp = np.concatenate((pp, np.ones((pp.shape[0],1))), axis=1)
        p3.append(pp)
    p = p3
    # print(p[0])
    p = [(T@pp.transpose()).transpose() for pp in p] # apply affine transfomatiton.
    p = [np.delete(pp, 2, axis=1) for pp in p]     # --- remove third dimension
    return p



def _reflect(p, theta=math.pi/2): # TODO: reflect should also be usable with repeat.
    # reflection over line thru origin
    # first rotate p by -theta, then reflect across y axis, then unrotate (by +theta)
    # y axis would be theta = pi/2
    th = theta - math.pi/2

    p = transform(p, theta=-th)
    T = np.array([[-1., 0.], [0., 1.]])
    p = [np.matmul(T, pp.transpose()).transpose() for pp in p]
    p = transform(p, theta=th)
    
    return p

# =========== FUNCTIONS ON PRIMTIVES.
def _repeat(p, N, T):
    p_out = []
    for i in range(N):
        if i>0:
            p = _tform(p, T) # apply transformation
        pthis = [np.copy(pp) for pp in p] # copy current state, and append
        p_out.extend(pthis)
    return p_out


def _connect(p1, p2):
    #  takes two primitives and makes a new one
    return p1 + p2


# ========== STROKES 
_line = [np.array([(0., 0.), (1., 0.)])] # --- unit line, from 0 to 1
_circle = [np.array([(0.5*math.cos(theta), 0.5*math.sin(theta)) for theta in np.linspace(0., 2.*math.pi, num=30)])] # --- circle, centered at 0, diameter 1

# --- regular polygons
if False:
    N = range(3,7)
    y = [(1/2)*(1)*(1/tan(pi/n)) for n in N]
    [plot(repeat(transform(line, x=-0.5, y=y[i]), n, makeAffine(theta=2*pi/n))) for i,n in enumerate(N)]

# ============= NOT PRIMITIVES.
def transform(p, s=1., theta=0., x=0., y=0., order="trs"):
            # order is one of the 6 ways you can permutate the three transformation primitives. 
            # write as a string (e.g. "trs" means scale, then rotate, then tranlate.)
            # input and output types guarantees a primitive will only be transformed once.

    T = _makeAffine(s, theta, x, y, order) # get affine matrix.
    p = _tform(p, T)
    return p
        

def savefig(p, fname="tmp.png"):
	ax = plot(p)
	ax.get_figure().savefig(fname)
	print("saved: {}".format(fname))

def plot(p):
    fig = plt.figure(figsize=(XYLIM,XYLIM))
    ax = fig.add_axes([-0.03, -0.03, 1.06, 1.06])
    ax.set_xlim(-XYLIM,XYLIM)
    ax.set_ylim(-XYLIM,XYLIM)
    [ax.plot(x[:,0], x[:,1], "-k") for x in p]
    return ax


def plotOnAxes(p, ax):
	ax.set_xlim(-XYLIM,XYLIM)
	ax.set_ylim(-XYLIM,XYLIM)

	[ax.plot(x[:,0], x[:,1], "-k") for x in p]
	return ax


def __fig2pixel(p, plotPxl=False, smoothing=0.):
#	smoothing is std of gaussian 2d filter. set to 0 to not smooth.
# 	https://stackoverflow.com/questions/43363388/how-to-save-a-greyscale-matplotlib-plot-to-numpy-array
    ax = plot(p)
    fig = ax.get_figure()
    fig.canvas.draw()
    ax.axis("off")

    width, height = fig.get_size_inches() * fig.get_dpi()
    # print("dpi: {}".format(fig.get_dpi()))
    # import pdb
    # pdb.set_trace()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    img = color.rgb2gray(img)

    if smoothing>0:
        img = gf(img, smoothing)
        
    if plotPxl:
        # - show the figure
        plt.figure()
        # plt.imshow(img, vmin=0, vmax=1, cmap="gray", interpolation="bicubic")
        plt.imshow(img, vmin=0, vmax=1, cmap="gray")

    return img

def __loss(p1, p2, plotPxl=False, smoothing=2):
    # loss function (compare two images)

    img1 = __fig2pixel(p1, plotPxl=plotPxl, smoothing=smoothing)
    img2 = __fig2pixel(p2, plotPxl=plotPxl, smoothing=smoothing)

    return np.linalg.norm(img2-img1)


def prog2pxl(p, WHdraw = 2*XYLIM):
    # takes a list of np array and outputs one pixel image
    # WHdraw, the size of drawing canvas (e.g. 6, if is xlim -3 to 3)
    
    # 1) create canvas
    WH = 128
    scale = WH/WHdraw
    data = np.zeros((WH, WH), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(data, cairo.Format.A8, WH-2, WH-2)

    # 2) create context
    context = cairo.Context(surface)
    # context.set_line_width(STROKESIZE)
    context.set_source_rgb(256,256,256)

    # 3) add each primitive
    for pp in p:
        pthis = pp
        
        pthis= pthis + WHdraw/2 # -- center
        pthis = pthis*scale # -- scale so that coordinates match

        for ppp in pthis:
            context.line_to(ppp[0], ppp[1])
        context.stroke() # this draws and also clears context.


    # 4) render
#     data = np.flip(data, 0)/255.0
    # data = data/255.0
    # surface.write_to_png("/tmp/test.png")
#     from matplotlib import pyplot as plt
#     plt.figure(figsize=(3,3))
#     # plt.xlim(-3,3)
#     # plt.ylim(-3,3)
#     plt.imshow(data, vmin=0, vmax=1, cmap="gray")
#     plt.savefig("/tmp/test.svg")

    return np.flip(data, 0)/255.0


def loss_pxl(img1, img2):
    return np.linalg.norm(img2-img1)
