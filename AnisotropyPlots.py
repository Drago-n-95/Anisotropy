# playing with equal area plots
from code import interact

import mplstereonet.stereonet_math
from matplotlib import pyplot as plt, widgets
import numpy as np
import pandas as pd
import scipy
import mplstereonet as mpls
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.linalg import eig
import nets
from numpy import sin, cos, sqrt, arctan, arcsin, arccos
from src.munmagtools.plotting.mpl import plotEA, inc2radius
from src.munmagtools.anisotropy.tensorfit import Eigenval2AnisoTensor, AnisotropyTensor2Results, CalcEigenValVec
from src.munmagtools.transforms import RVec, RotateTensor
from jelinekstat.jelinekstat import tensorStat
from jelinekstat.tools import proyAllEllipses2LongLat, confRegions2PPlanes, splitIterables


def find_my_means(inc, dec):
    # Takes inclinations and declinations of a set of points and returns the incnliation and declination of its mean, R, k and alpha95
    
    N = len(dec)
    suml = 0
    summ = 0
    sumn = 0

    for j in range(N):
        # goes through all the inclinations and declinations and turns them into radians
        radi = np.radians(inc[j])
        radd = np.radians(dec[j])
        # Implements formula 6.12 from Butler
        li = cos(radi) * cos(radd)
        suml += li
        mi = cos(radi) * sin(radd)
        summ += mi
        ni = sin(radi)
        sumn += ni

    # Formula 6.13 and 6.14 form Butler
    R = sqrt(suml ** 2 + summ ** 2 + sumn ** 2)
    l = suml / R
    m = summ / R
    n = sumn / R
    k = (N-1) / (N-R)

    # Formula 6.15 from Butler
    Dm = arctan(m / l)
    Im = arcsin(n)

    # Formula 6.21 from Butler
    alpha95 = arccos(1 - ((N - R) / R) * ((1 / 0.05) ** (1 / (N - 1)) - 1))

    return Im, Dm, alpha95, R, k

# create some normal distributed inclinations and declinations
decli = np.random.normal(45, 10, size=20)   # declinations
incli = np.random.normal(45, 10, size=20)   # inclinations

Im, Dm, a95, R, k = find_my_means(incli, decli)

# show clustered directions
fig = plotEA([list(decli), ], [list(incli), ], lower=False, title='random clustered directions')
# @Drago would be nice to plot here an error circle (e.g. a95)
ima = inc2radius(np.degrees(Im))
# circle of confidence
gamma = np.degrees(a95)
pole = [90. - np.degrees(Dm), np.degrees(Im)] #poles

plt.plot(np.radians(90. - np.degrees(Dm)), ima, 'ro')

#for pole in poles: # step through the list of poles
points = nets.pts_on_a_plane(pole,gamma) # get the points gamma away from this pole
for pt in points:
   if pt[1]<0: # upper hemisphere
      plt.plot(np.radians(pt[0]), nets.EqualArea(-pt[1]), 'g.', markersize=0.5)
   else: # lower hemisphere
      plt.plot(np.radians(pt[0]), nets.EqualArea(pt[1]), 'b.', markersize=0.5)


plt.show()


# plot some anisotropy data
# create random tensors
tensors = []
aniso_dicts = []
'''
for i in range(100):
    tensors.append(RotateTensor(Eigenval2AnisoTensor([1.1, 1, 0.9]), scipy.spatial.transform.Rotation.from_rotvec(RVec(40)).as_matrix()))
    aniso_dict = {}
    AnisotropyTensor2Results(tensors[-1], aniso_dict)
    aniso_dicts.append(aniso_dict)
'''
csv_file_path = '/home/dragomir/Documents/test.csv'

# Read CSV file into a DataFrame
res = pd.read_csv(csv_file_path)
#res = pd.DataFrame(aniso_dicts)
d = res[['D1p', 'D2p', 'D3p']].values.tolist()
i = res[['I1p', 'I2p', 'I3p']].values.tolist()
#res.to_csv('/home/dragomir/Documents/test.csv', index=False)
dt_red = res.loc[:, "D3p"]
it_red = res.loc[:, "I3p"]
dt_blue = res.loc[:, "D2p"]
it_blue = res.loc[:, "I2p"]
dt_green = res.loc[:, "D1p"]
it_green = res.loc[:, "I1p"]

Rmatrix = res.loc[:, "R"]

Kmatrix = np.ndarray(shape=(100, 6), dtype=float)
Kmatrix = []
print(np.shape(Kmatrix))
for i in range(len(Rmatrix)):
    Klist = [Rmatrix[i][0][0], Rmatrix[i][1][1], Rmatrix[i][2][2], Rmatrix[i][0][1],
             Rmatrix[i][1][2], Rmatrix[i][0][2]]
    Kmatrix.append(Klist)
#fig = plotEA(d, i, is_aniso=[True, ]*len(d))
print("Here we have a Kmatrix: \n")
print(Kmatrix)
green_incs_for_mean = it_green.copy()
green_decs_for_mean = dt_green.copy()

red_incs_for_mean = it_red.copy()
red_decs_for_mean = dt_red.copy()

blue_incs_for_mean = it_blue.copy()
blue_decs_for_mean = dt_blue.copy()

for k in range(100):
    if dt_green[k] > 90. and dt_green[k] < 270.:
        green_incs_for_mean[k] = it_green[k] * (-1)
        green_decs_for_mean[k] = dt_green[k] - 180.
    else:
        green_incs_for_mean[k] = it_green[k]
        green_decs_for_mean[k] = dt_green[k]

for k in range(100):
    if dt_blue[k] > 180.:
        blue_incs_for_mean[k] = it_blue[k] * (-1)
        blue_decs_for_mean[k] = dt_blue[k] - 180.
    else:
        blue_incs_for_mean[k] = it_blue[k]
        blue_decs_for_mean[k] = dt_blue[k]


# Create a stereonet with grid lines.
fig, ax = mpls.subplots(figsize=(9, 6))
ax.grid(color='k', alpha=0.2)
ax.line(it_red, dt_red, 'ro', alpha=0.5)
ax.line(it_blue, dt_blue, 'bs', alpha=0.5)
ax.line(it_green, dt_green, 'g^', alpha=0.5)

vector_green, stats_green = mpls.find_fisher_stats(green_incs_for_mean, green_decs_for_mean, conf=0.95)
meanGreenInc, meanGreenDec, a95_green, R_green, k_green = find_my_means(green_incs_for_mean, green_decs_for_mean)
ax.line(np.degrees(meanGreenInc), np.degrees(meanGreenDec), 'k^', markersize=10)
#ax.cone(np.degrees(meanGreenInc), np.degrees(meanGreenDec), np.degrees(stats_green[1]), facecolor="None")
#print(np.degrees(meanGreenInc), np.degrees(meanGreenDec), np.degrees(a95_green))
#print(vector_green, np.degrees(stats_green[1]))

vector_red, stats_red = mpls.find_fisher_stats(red_incs_for_mean, red_decs_for_mean, conf=0.95)
meanRedInc, meanRedDec, a95_red, R_red, k_red = find_my_means(it_red, dt_red)
ax.line(np.degrees(meanRedInc), np.degrees(meanRedDec), 'ko', markersize=10)
#ax.cone(np.degrees(meanRedInc), np.degrees(meanRedDec), np.degrees(stats_red[1]), facecolor="None")
#print(np.degrees(meanRedInc), np.degrees(meanRedDec), np.degrees(a95_red))
#print(vector_red, np.degrees(stats_red[1]))

vector_blue, stats_blue = mpls.find_fisher_stats(blue_incs_for_mean, blue_decs_for_mean, conf=0.95)
meanBlueInc, meanBlueDec, a95_blue, R_blue, k_blue = find_my_means(blue_incs_for_mean, blue_decs_for_mean)
ax.line(np.degrees(meanBlueInc), np.degrees(meanBlueDec), 'ks', markersize=10)
#ax.cone(np.degrees(meanBlueInc), np.degrees(meanBlueDec), np.degrees(stats_blue[1]), facecolor="None")
#print(np.degrees(meanBlueInc), np.degrees(meanBlueDec), np.degrees(stats_blue[1]))
#print(vector_blue, np.degrees(a95_blue))

'''
Ps = ['p1', 'p2', 'p3']
markers = ['ks', 'k^', 'ko']
MajorAxis, MinorAxis, Theta = [], [], []

for k in range(3):
    MajorAxis.append(jelinekStatsSummary[Ps[k], 'majAx'])
    MinorAxis.append(jelinekStatsSummary[Ps[k], 'minAx'])
    Theta.append(jelinekStatsSummary[Ps[k], 'incl'])

k = jelinekStatsSummary['k']
x, y, PPlanePlots = confRegions2PPlanes(MajorAxis, MinorAxis, Theta, True, 0.95)
kRegionsLong, kRegionsLat = proyAllEllipses2LongLat(x, y, k)

for i in range(3):
    Plg = jelinekStatsSummary[Ps[i], 'plg']
    Trd = jelinekStatsSummary[Ps[i], 'trd']
    #ax.line(np.degrees(mean_green_inc), np.degrees(mean_green_dec + np.pi), 'k^', markersize=5)
    kRegionsLongSplitted, kRegionsLatSplitted = splitIterables(
        kRegionsLong[i], kRegionsLat[i])
    for j in range(len(kRegionsLongSplitted)):
        #print((kRegionsLongSplitted[j], kRegionsLatSplitted[j]))
        ax.plot(kRegionsLongSplitted[j], kRegionsLatSplitted[j], 'k', lw=1)
'''


#Normalize K matrix
NormKmatrix = np.empty([100, 6])
SumKmatrix = np.zeros([1, 6])

for i in range(100):
    for k in range(6):
        NormKmatrix[i][k] = Kmatrix[i][k] / (Kmatrix[i][0] + Kmatrix[i][1] + Kmatrix[i][2])

    SumKmatrix += NormKmatrix[i]

MeanTensorK = SumKmatrix / 6
T = np.empty([3, 3])
T[0, 0], T[1, 1], T[2, 2] = MeanTensorK[0, 0], MeanTensorK[0, 1], MeanTensorK[0, 2]
T[0, 1], T[1, 0] = MeanTensorK[0, 3], MeanTensorK[0, 3]
T[1, 2], T[2, 1] = MeanTensorK[0, 4], MeanTensorK[0, 4]
T[2, 0], T[0, 2] = MeanTensorK[0, 5], MeanTensorK[0, 5]

eigenvalue, eigenvector = eig(T)


def CalcCovMatrix(NormMat, MeanTens):
    N = len(NormMat)
    CM = np.zeros([6, 6])
    CM00, CM11, CM22, CM33, CM44, CM55 = 0, 0, 0, 0, 0, 0
    CM01, CM12, CM20, CM30, CM31, CM32 = 0, 0, 0, 0, 0, 0
    CM40, CM41, CM42, CM43 = 0, 0, 0, 0
    CM50, CM51, CM52, CM53, CM54 = 0, 0, 0, 0, 0
    for i in range(N):
        CM00 += (NormMat[i, 0] - MeanTens[0, 0]) ** 2
        CM11 += (NormMat[i, 1] - MeanTens[0, 1]) ** 2
        CM22 += (NormMat[i, 2] - MeanTens[0, 2]) ** 2
        CM33 += (NormMat[i, 3] - MeanTens[0, 3]) ** 2
        CM44 += (NormMat[i, 4] - MeanTens[0, 4]) ** 2
        CM55 += (NormMat[i, 5] - MeanTens[0, 5]) ** 2
        CM01 += (NormMat[i, 0] - MeanTens[0, 0]) * (NormMat[i, 1] - MeanTens[0, 1])
        CM12 += (NormMat[i, 1] - MeanTens[0, 1]) * (NormMat[i, 2] - MeanTens[0, 2])
        CM20 += (NormMat[i, 2] - MeanTens[0, 2]) * (NormMat[i, 0] - MeanTens[0, 0])
        CM30 += (NormMat[i, 3] - MeanTens[0, 3]) * (NormMat[i, 0] - MeanTens[0, 0])
        CM31 += (NormMat[i, 3] - MeanTens[0, 3]) * (NormMat[i, 1] - MeanTens[0, 1])
        CM32 += (NormMat[i, 3] - MeanTens[0, 3]) * (NormMat[i, 2] - MeanTens[0, 2])
        CM40 += (NormMat[i, 4] - MeanTens[0, 4]) * (NormMat[i, 0] - MeanTens[0, 0])
        CM41 += (NormMat[i, 4] - MeanTens[0, 4]) * (NormMat[i, 1] - MeanTens[0, 1])
        CM42 += (NormMat[i, 4] - MeanTens[0, 4]) * (NormMat[i, 2] - MeanTens[0, 2])
        CM43 += (NormMat[i, 4] - MeanTens[0, 4]) * (NormMat[i, 3] - MeanTens[0, 3])
        CM50 += (NormMat[i, 5] - MeanTens[0, 5]) * (NormMat[i, 0] - MeanTens[0, 0])
        CM51 += (NormMat[i, 5] - MeanTens[0, 5]) * (NormMat[i, 1] - MeanTens[0, 1])
        CM52 += (NormMat[i, 5] - MeanTens[0, 5]) * (NormMat[i, 2] - MeanTens[0, 2])
        CM53 += (NormMat[i, 5] - MeanTens[0, 5]) * (NormMat[i, 3] - MeanTens[0, 3])
        CM54 += (NormMat[i, 5] - MeanTens[0, 5]) * (NormMat[i, 4] - MeanTens[0, 4])


    CM[0, 0], CM[1, 1], CM[2, 2], CM[3, 3], CM[4, 4], CM[5, 5] = CM00*(N-1), CM11*(N-1), CM22*(N-1), CM33*(N-1), CM44*(N-1), CM55*(N-1)
    CM[0, 1], CM[1, 0] = CM01 * (N-1), CM01 * (N-1)
    CM[1, 2], CM[2, 1] = CM12 * (N-1), CM12 * (N-1)
    CM[2, 0], CM[0, 2] = CM20 * (N-1), CM20 * (N-1)
    CM[3, 0], CM[0, 3] = CM30 * (N-1), CM30 * (N-1)
    CM[3, 1], CM[1, 3] = CM31 * (N-1), CM31 * (N-1)
    CM[3, 2], CM[2, 3] = CM32 * (N-1), CM32 * (N-1)
    CM[4, 0], CM[0, 4] = CM40 * (N-1), CM40 * (N-1)
    CM[4, 1], CM[1, 4] = CM41 * (N-1), CM41 * (N-1)
    CM[4, 2], CM[2, 4] = CM42 * (N-1), CM42 * (N-1)
    CM[4, 3], CM[3, 4] = CM43 * (N-1), CM43 * (N-1)
    CM[5, 0], CM[0, 5] = CM50 * (N-1), CM50 * (N-1)
    CM[5, 1], CM[1, 5] = CM51 * (N-1), CM51 * (N-1)
    CM[5, 2], CM[2, 5] = CM52 * (N-1), CM52 * (N-1)
    CM[5, 3], CM[3, 5] = CM53 * (N-1), CM53 * (N-1)
    CM[5, 4], CM[4, 5] = CM54 * (N-1), CM54 * (N-1)

    return CM

CovarMatrix = CalcCovMatrix(NormKmatrix, MeanTensorK)

def RotateMatrix(eigenvec):
    RotMatrix = np.empty([6, 6])
    for k in range(3):
        RotMatrix[k][k] = eigenvec[k][k] ** 2
    RotMatrix[0, 1], RotMatrix[1, 0] = eigenvec[1, 0] ** 2, eigenvec[0, 1] ** 2
    RotMatrix[0, 2], RotMatrix[2, 0] = eigenvec[2, 0] ** 2, eigenvec[0, 2] ** 2
    RotMatrix[1, 2], RotMatrix[2, 1] = eigenvec[2, 1] ** 2, eigenvec[1, 2] ** 2
    RotMatrix[0, 3], RotMatrix[0, 4], RotMatrix[0, 5] = 2*(eigenvec[0, 0]*eigenvec[1, 0]), 2*(eigenvec[1, 0]*eigenvec[2, 0]), 2*(eigenvec[2, 0]*eigenvec[0, 0])
    RotMatrix[1, 3], RotMatrix[1, 4], RotMatrix[1, 5] = 2*(eigenvec[0, 1]*eigenvec[1, 1]), 2*(eigenvec[1, 1]*eigenvec[2, 1]), 2*(eigenvec[2, 1]*eigenvec[0, 1])
    RotMatrix[2, 3], RotMatrix[2, 4], RotMatrix[2, 5] = 2*(eigenvec[0, 2]*eigenvec[1, 2]), 2*(eigenvec[1, 2]*eigenvec[2, 2]), 2*(eigenvec[2, 2]*eigenvec[0, 2])
    RotMatrix[3, 0], RotMatrix[3, 1], RotMatrix[3, 2] = eigenvec[0, 0]*eigenvec[0, 1], eigenvec[1, 0]*eigenvec[1, 1], eigenvec[2, 0]*eigenvec[2, 1]
    RotMatrix[4, 0], RotMatrix[4, 1], RotMatrix[4, 2] = eigenvec[0, 1]*eigenvec[0, 2], eigenvec[1, 1]*eigenvec[1, 2], eigenvec[2, 1]*eigenvec[2, 2]
    RotMatrix[5, 0], RotMatrix[5, 1], RotMatrix[5, 2] = eigenvec[0, 2]*eigenvec[0, 0], eigenvec[1, 2]*eigenvec[1, 0], eigenvec[2, 2]*eigenvec[2, 0]
    RotMatrix[3, 3], RotMatrix[3, 4], RotMatrix[3, 5] = eigenvec[0, 0] * eigenvec[1, 1] + eigenvec[1, 0] * eigenvec[0, 1], eigenvec[1, 0] * eigenvec[2, 1] + eigenvec[2, 0] * eigenvec[1, 1], eigenvec[2, 0] * eigenvec[0, 1] + eigenvec[0, 0] * eigenvec[2, 1]
    RotMatrix[4, 3], RotMatrix[4, 4], RotMatrix[4, 5] = eigenvec[0, 1] * eigenvec[1, 2] + eigenvec[1, 1] * eigenvec[0, 2], eigenvec[1, 1] * eigenvec[2, 2] + eigenvec[2, 1] * eigenvec[1, 2], eigenvec[2, 1] * eigenvec[0, 2] + eigenvec[0, 1] * eigenvec[2, 2]
    RotMatrix[5, 3], RotMatrix[5, 4], RotMatrix[5, 5] = eigenvec[0, 2] * eigenvec[1, 0] + eigenvec[1, 2] * eigenvec[0, 0], eigenvec[1, 2] * eigenvec[2, 0] + eigenvec[2, 2] * eigenvec[1, 0], eigenvec[2, 2] * eigenvec[0, 0] + eigenvec[0, 2] * eigenvec[2, 0]

    return RotMatrix

RotationMatrix = RotateMatrix(eigenvector)
RotationMatrix_transpose = RotationMatrix.T
CovarMatrix_rotated = RotationMatrix @ CovarMatrix @ RotationMatrix_transpose

W1 = np.empty([2, 2])
W2 = np.empty([2, 2])
W3 = np.empty([2, 2])
W1[0, 0], W1[0, 1], W1[1, 0], W1[1, 1] = CovarMatrix_rotated[3, 3]/(eigenvalue[0] - eigenvalue[1])**2, CovarMatrix_rotated[3, 5]/((eigenvalue[0] - eigenvalue[1])*(eigenvalue[2] - eigenvalue[0])), CovarMatrix_rotated[3, 5]/((eigenvalue[0] - eigenvalue[1])*(eigenvalue[2] - eigenvalue[0])), CovarMatrix_rotated[5, 5]/(eigenvalue[2] - eigenvalue[0])**2

W2[0, 0], W2[0, 1], W2[1, 0], W2[1, 1] = CovarMatrix_rotated[4, 4]/(eigenvalue[1] - eigenvalue[2])**2, CovarMatrix_rotated[4, 3]/((eigenvalue[1] - eigenvalue[2])*(eigenvalue[0] - eigenvalue[1])), CovarMatrix_rotated[4, 3]/((eigenvalue[1] - eigenvalue[2])*(eigenvalue[0] - eigenvalue[1])), CovarMatrix_rotated[3, 3]/(eigenvalue[0] - eigenvalue[1])**2

W3[0, 0], W3[0, 1], W3[1, 0], W3[1, 1] = CovarMatrix_rotated[5, 5]/(eigenvalue[2] - eigenvalue[0])**2, CovarMatrix_rotated[5, 4]/((eigenvalue[2] - eigenvalue[0])*(eigenvalue[1] - eigenvalue[2])), CovarMatrix_rotated[5, 4]/((eigenvalue[2] - eigenvalue[0])*(eigenvalue[1] - eigenvalue[2])), CovarMatrix_rotated[4, 4]/(eigenvalue[1] - eigenvalue[2])**2

W1_eigenval, W1_eigenvec = eig(W1)
W2_eigenval, W2_eigenvec = eig(W2)
W3_eigenval, W3_eigenvec = eig(W3)

W_eigenvals = [W1_eigenval, W2_eigenval, W3_eigenval]
W_eigenvectors = [W1_eigenvec, W2_eigenvec, W3_eigenvec]

def ConfEllipses(eigenvalues_W, eigenvectors_W, numTensors):

    from scipy.stats import f

    majorAxis = []
    minorAxis = []
    theta = []
    stat = f.ppf(0.95, 2, 4)  # F distribution
    print(stat)
    print(numTensors)
    F = stat * 2 * (numTensors - 1) / (numTensors * (numTensors - 2))
    for i in range(3):
        majorAxis.append(np.arctan(np.sqrt(F * eigenvalues_W[i][0])))
        minorAxis.append(arctan(np.sqrt(F * eigenvalues_W[i][1])))
        theta.append(np.arctan(eigenvectors_W[i][1, 0] / eigenvectors_W[i][0, 0]))


    return np.array(majorAxis), np.array(minorAxis), np.array(theta)

majAxis, minAxis, Theta = ConfEllipses(W_eigenvals, W_eigenvectors, len(Kmatrix))
# Coordiantes of the three ellipses in each P-plane.
x, y, PPlanePlots = confRegions2PPlanes(majAxis, minAxis, np.degrees(Theta), True, 0.95)


# (lon, lat) notation of each confidence region.
kRegionsLong, kRegionsLat = proyAllEllipses2LongLat(x, y, MeanTensorK[0])

# Confidence regions
for i in range(3):
    kRegionsLongSplitted, kRegionsLatSplitted = splitIterables(kRegionsLong[i], kRegionsLat[i])
    for j in range(len(kRegionsLongSplitted)):
        ax.plot(kRegionsLongSplitted[j], kRegionsLatSplitted[j], ':k', lw=1)

print(Rmatrix[0])
plt.show()
'''
instr_inc = np.array([90, 90, 85.5, 55, 61, 51, 5, 80, 45, 76, 46.5, 79.5, 40, 79.5, 35, 79, 35, 10])
instr_dec = np.array([330, 300, 253, 291, 342, 247, 64, 210, 213, 159, 142, 256, 251, 14, 362, 97, 73, 45])
app_dec = np.array([329.3, 298.7, 256.1, 293.4, 343.9, 246.6, 67, 216.4, 210.4, 158.6, 141.1, 253.2, 252.1, 12.9, 358.7, 97.5, 74.5, 48.8])
app_inc = np.array([90.5, 89.6, 85.7, 55.4, 63.6, 51.9, 4.8, 81, 45, 76.1, 47.1, 79.5, 39.9, 79.6, 34.4, 79.1, 36, 10.4])


ries_dec_green = [185.4, 214.7, 180, 181.7, 180.3, 185.7, 184.9, 188.4, 177.8, 184.1]
ries_inc_green = [46.4, 54.7, 62.9, 63, 64.5, 61.9, 65.4, 62.1, 62.6, 64.2]

ries_dec_blue = [179.5, 177.2, 185.8, 184, 188.6]
ries_inc_blue = [48.7, 65.7, 64.4, 66.3, 63.3]

ries_dec_red = [182.2, 200.4, 177.6, 189.9, 186.8, 186.6, 181.8, 192.5]
ries_inc_red = [56.6, 65.2, 66.8, 62.2, 64.2, 60.6, 64.9, 64.2]

fig, ax = mpls.subplots(figsize=(9, 6))
ax.grid(color='k', alpha=0.2)
ax.line(ries_inc_green, ries_dec_green, 'go')
ax.line(ries_inc_blue, ries_dec_blue, 'bo')
ax.line(ries_inc_red, ries_dec_red, 'ro')
#ax.line(app_inc, app_dec, 'g^')

plt.show()
'''
'''
plt.subplot(211)
plt.scatter(instr_dec, app_dec)
plt.ylabel('Sensor Declination')
#plt.xlabel('Instrument Declination')

plt.subplot(212)
plt.scatter(instr_inc, app_inc)
plt.ylabel('Sensor Inclination')
plt.xlabel('Instrument Inclination')

diff_decs = app_dec - instr_dec
diff_incs = app_inc - instr_inc
med_decs = np.median(diff_decs)
med_incs = np.median(diff_incs)
std_decs = np.std(diff_decs)
std_incs = np.std(diff_incs)
print(med_incs, med_decs)
print(std_incs, std_decs)

plt.show()
'''

