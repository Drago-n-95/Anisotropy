import pandas as pd
import os
from numpy import arctan, arcsin, sqrt, cos, sin, arccos, arctan2
import numpy as np
import mplstereonet as mpls
import matplotlib.pyplot as plt


def find_my_means(Inc, Dec):
    # Takes inclinations and declinations of a set of points and returns the incnliation and declination of its mean, R, k and alpha95

    N = len(Dec)
    suml = 0
    summ = 0
    sumn = 0

    for j in range(N):

        # Implements formula 6.12 from Butler
        li = cos(Inc[j]) * cos(Dec[j])
        suml += li
        mi = cos(Inc[j]) * sin(Dec[j])
        summ += mi
        ni = sin(Inc[j])
        sumn += ni

    # Formula 6.13 and 6.14 form Butler
    R = sqrt(suml ** 2 + summ ** 2 + sumn ** 2)
    l = suml / R
    m = summ / R
    n = sumn / R
    k = (N - 1) / (N - R)

    # Formula 6.15 from Butler
    Dm = arctan(m/l)
    Im = arcsin(n)

    # Formula 6.21 from Butler
    argum = 1 - ((N - R) / R) * ((1 / 0.05) ** (1 / (N - 1)) - 1)
    if argum < -1:
        print(R)
    alpha95 = arccos(argum)

    return Im, Dm, alpha95, R, k, N


cur_path = os.path.dirname(__file__)

column_names = ["sample", "Xc", "Yc", "Zc", "MAG[A/m]", "Dg", "Ig", "Ds", "Is", "alpha95"]

filename = os.path.relpath('..//..//ARM_final_data//corefldlst.csv', cur_path)

df_names = pd.read_csv(filename, sep=',')
df_names.columns = ["sample", "", "", "", "aniso_percentage", "P", "L", "F"]

D_ref = [45, 135, 225, 315, 90, 180, 270, 360]
I_ref = [0, 0, 0, 0, 45, 45, 45, 45]

k_R_values = []
NRM92_ARM_values = []
dispersion_values = []
R_values = []
k_values = []
x_17, x_28, x_39, x_410, x_511, x_612 = [], [], [], [], [], []
y_17, y_28, y_39, y_410, y_511, y_612 = [], [], [], [], [], []
z_17, z_28, z_39, z_410, z_511, z_612 = [], [], [], [], [], []

#Difference between the maximum distance from the mean value and the dispersion
diff_dispersion_value = []

for j in range(len(df_names['sample'])):
    #Taking AARM information
    filepath = "..//..//ARM_final_data//" + df_names['sample'][j] + "B.pmd"
    filename = os.path.relpath(filepath, cur_path)

    df = pd.read_csv(filename, sep='\s+', skiprows=3)
    df.columns = column_names

    #Taking demag information
    filepath = "..//..//AF_demag_Sijan_2022//" + df_names['sample'][j] + "B.pmd"
    filename = os.path.relpath(filepath, cur_path)
    column_names_demag = ["sample", "Xc", "Yc", "Zc", "MAG[A/m]", "Dg", "Ig", "Ds", "Is", "alpha95", "", "", ""]
    df_AF_demag = pd.read_csv(filename, sep='\s+', skiprows=2)
    df_AF_demag.columns = column_names_demag

    NRM_demag = df_AF_demag['MAG[A/m]']
    NRM_sample = NRM_demag[0]

    Xc = df['Xc']
    Yc = df['Yc']
    Zc = df['Zc']

    Xc_mean, Yc_mean, Zc_mean,  = [], [], []
    Xc_NRM, Yc_NRM, Zc_NRM = [], [], []
    
    for i in range(6):
        Xc_mean.append((Xc[i+6] - Xc[i]) / 2)
        Yc_mean.append((Yc[i+6] - Yc[i]) / 2)
        Zc_mean.append((Zc[i+6] - Zc[i]) / 2)

        Xc_NRM.append((Xc[i+6] + Xc[i]) / 2)
        Yc_NRM.append((Yc[i+6] + Yc[i]) / 2)
        Zc_NRM.append((Zc[i+6] + Zc[i]) / 2)

        if i == 0:
            x_17.append((Xc[i+6] + Xc[i]) / 2)
            y_17.append((Yc[i + 6] + Yc[i])/2)
            z_17.append((Zc[i + 6] + Zc[i])/2)

        elif i == 1:
            x_28.append((Xc[i + 6] + Xc[i]) / 2)
            y_28.append((Yc[i + 6] + Yc[i]) / 2)
            z_28.append((Zc[i + 6] + Zc[i]) / 2)

        elif i == 2:
            x_39.append((Xc[i + 6] + Xc[i]) / 2)
            y_39.append((Yc[i + 6] + Yc[i]) / 2)
            z_39.append((Zc[i + 6] + Zc[i]) / 2)

        elif i == 3:
            x_410.append((Xc[i + 6] + Xc[i]) / 2)
            y_410.append((Yc[i + 6] + Yc[i]) / 2)
            z_410.append((Zc[i + 6] + Zc[i]) / 2)

        elif i == 4:
            x_511.append((Xc[i + 6] + Xc[i]) / 2)
            y_511.append((Yc[i + 6] + Yc[i]) / 2)
            z_511.append((Zc[i + 6] + Zc[i]) / 2)

        elif i == 5:
            x_612.append((Xc[i + 6] + Xc[i]) / 2)
            y_612.append((Yc[i + 6] + Yc[i]) / 2)
            z_612.append((Zc[i + 6] + Zc[i]) / 2)

    
    I, D = [], []
    I_NRM, D_NRM = [], []
    
    for i in range(6):
        B = sqrt(Xc_mean[i] ** 2 + Yc_mean[i] ** 2 + Zc_mean[i] ** 2)
        D.append(arctan(Yc_mean[i] / Xc_mean[i]))
        I.append(arcsin(Zc_mean[i] / B))

        B_NRM = sqrt(Xc_NRM[i] ** 2 + Yc_NRM[i] ** 2 + Zc_NRM[i] ** 2)
        if Xc_NRM[i] == 0:
            Xc_NRM[i] = Xc_NRM[i] + 1*10**(-12)
        D_NRM.append(arctan(Yc_NRM[i] / Xc_NRM[i]))
        I_NRM.append(arcsin(Zc_NRM[i] / B_NRM))

    for i in range(len(I)):
        if D[i] < 0:
            D[i] = D[i] + 2 * np.pi

    for i in range(len(I)):
        if I[i] < 0:
            I[i] = I[i] * (-1)
            D[i] = D[i] + np.pi


    # Create a stereonet with grid lines.
    fig, ax = mpls.subplots(figsize=(9, 6))
    ax.grid(color='k', alpha=0.2)
    ax.line(np.degrees(I), np.degrees(D), 'k^', markersize=10)
    ax.line(I_ref, D_ref, 'ro', markersize=10)
    plt.title(df_names["sample"][j] + "  Level of anisotropy: " + str(np.floor(df_names["aniso_percentage"][j])) + "%")
    plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/SushiBarAnisoPlots/" + df_names["sample"][j] + "B.png")

    Im, Dm, a95, R, k, N = find_my_means(I_NRM, D_NRM)

    X = cos(Im)*cos(Dm)
    Y = cos(Im)*sin(Dm)
    Z = sin(Im)

    norm_mean = np.sqrt(X**2 + Y**2 + Z**2)

    list_delta_angle = []
    for jt in range(len(I_NRM)):
        norm_NRM_vec = np.sqrt(Xc_NRM[jt]**2 + Yc_NRM[jt]**2 + Zc_NRM[jt]**2)
        delta = arccos((X*Xc_NRM[jt] + Y*Yc_NRM[jt] + Z*Zc_NRM[jt]) / (norm_NRM_vec * norm_mean)) ** 2
        list_delta_angle.append(np.degrees(delta))

    disper = np.sqrt(np.sum(list_delta_angle) / 5)
    dispersion_values.append(disper)

    diff_dispersion_value.append(np.max(list_delta_angle) - disper)

    mag_NRM92 = []
    mag_ARM = []
    for y in range(len(Xc_NRM)):
        mag_ARM.append(np.sqrt(Xc_mean[y] ** 2 + Yc_mean[y] ** 2 + Zc_mean[y] ** 2) / (11 * 10 ** (-6)))
        mag_NRM92.append(np.sqrt(Xc_NRM[y] ** 2 + Yc_NRM[y] ** 2 + Zc_NRM[y] ** 2) / (11 * 10 ** (-6)))
    mean_NRM92 = np.sum(mag_NRM92) / len(mag_NRM92)
    mean_ARM = np.sum(mag_ARM) / len(mag_ARM)

    k_R_values.append(k / R)
    NRM92_ARM_values.append(mean_NRM92 / mean_ARM)
    R_values.append(R)
    k_values.append(k)

    fig, ax = mpls.subplots(figsize=(9, 6))
    ax.grid(color='k', alpha=0.2)

    step_colors = ["ko", "bo", "mo", "co", "go", "yo"]

    for it in range(len(I_NRM)):
        template = (u"{pair}\n")
        label = template.format(pair=str(it+1) + "-" + str(it+7))

        clr = step_colors[it]
        if np.degrees(I_NRM[it]) < 0:
            ax.line(np.degrees(I_NRM[it])*(-1), np.degrees(D_NRM[it]), clr, markersize=10, markerfacecolor='none', label=label)
            #legend1 = plt.legend(label, loc='best')
            #ax.add_artist(legend1)
        else:
            ax.line(np.degrees(I_NRM[it]), np.degrees(D_NRM[it]), clr, markersize=10, label=label)
            #legend1 = plt.legend(label, loc='best')
            #ax.add_artist(legend1)

    template = (u"Mean Vector P/B: {plunge:0.0f}\u00B0/{bearing:0.0f}\u00B0\n"
                u"\u03B1 95: {fisher:0.1f}\u00B0\n"
                u"R_value: {R:0.2f}\n"
                "k-Value: {k:0.1f}\n"
                "Dispersion: {disper:0.1f}\n"
                "NRM92/ARM: {NRM92_ARM:0.2f}\n"
                )

    label = template.format(plunge=np.degrees(Im), bearing=np.degrees(Dm), fisher=np.degrees(a95), R=R, k=k, disper=disper, NRM92_ARM=mean_NRM92/mean_ARM)
    if np.degrees(Im) < 0:
        ax.line(np.degrees(Im)*(-1), np.degrees(Dm), 'ro', markersize=10, markerfacecolor='none', label=label)
        ax.cone(np.degrees(Im)*(-1), np.degrees(Dm), np.degrees(a95), facecolor="None", edgecolor="black")
    else:
        ax.line(np.degrees(Im), np.degrees(Dm), 'ro', markersize=10, label=label)
        ax.cone(np.degrees(Im), np.degrees(Dm), np.degrees(a95), facecolor="None", edgecolor="black")

    plt.legend(loc="best")
    plt.title(df_names["sample"][j] + "  Degree of anisotropy: " + str(np.floor(df_names["aniso_percentage"][j])) + "%")
    plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/Aniso_error_estimation/" + df_names["sample"][j] + "B.png")


std_x_17, std_x_28, std_x_39, std_x_410, std_x_511, std_x_612 = np.std(x_17), np.std(x_28), np.std(x_39), np.std(x_410), np.std(x_511), np.std(x_612)
std_y_17, std_y_28, std_y_39, std_y_410, std_y_511, std_y_612 = np.std(y_17), np.std(y_28), np.std(y_39), np.std(y_410), np.std(y_511), np.std(y_612)
std_z_17, std_z_28, std_z_39, std_z_410, std_z_511, std_z_612 = np.std(z_17), np.std(z_28), np.std(z_39), np.std(z_410), np.std(z_511), np.std(z_612)

mean_x_17, mean_x_28, mean_x_39, mean_x_410, mean_x_511, mean_x_612 = np.mean(x_17), np.mean(x_28), np.mean(x_39), np.mean(x_410), np.mean(x_511), np.mean(x_612)
mean_y_17, mean_y_28, mean_y_39, mean_y_410, mean_y_511, mean_y_612 = np.mean(y_17), np.mean(y_28), np.mean(y_39), np.mean(y_410), np.mean(y_511), np.mean(y_612)
mean_z_17, mean_z_28, mean_z_39, mean_z_410, mean_z_511, mean_z_612 = np.mean(z_17), np.mean(z_28), np.mean(z_39), np.mean(z_410), np.mean(z_511), np.mean(z_612)

means = [mean_x_17, mean_y_17, mean_z_17, mean_x_28, mean_y_28, mean_z_28, mean_x_39, mean_y_39, mean_z_39, mean_x_410, mean_y_410, mean_z_410, mean_x_511, mean_y_511, mean_z_511, mean_x_612, mean_y_612, mean_z_612]
stds = [std_x_17, std_y_17, std_z_17, std_x_28, std_y_28, std_z_28, std_x_39, std_y_39, std_z_39, std_x_410, std_y_410, std_z_410, std_x_511, std_y_511, std_z_511, std_x_612, std_y_612, std_z_612]
bins = [0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75, 4.0, 4.25, 4.75, 5.0, 5.25, 5.75, 6.0, 6.25]

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(x=dispersion_values, y=NRM92_ARM_values)
plt.xlabel("S")
plt.ylabel("NRM92/ARM")
#ax.scatter(x=R_values, y=dispersion_values)
#ax.scatter(x=k_values, y=dispersion_values)
#ax.scatter(x=R_values, y=NRM92_ARM_values)    plt.show()

'''
ax.scatter(bins, means)
my_xticks = ['x', 'y\n1-7 pairs', 'z', 'x', 'y\n2-8 pairs', 'z', 'x', 'y\n3-9 pairs', 'z', 'x', 'y\n4-10 pairs', 'z', 'x', 'y\n5-11 pairs', 'z', 'x', 'y\n6-12 pairs', 'z']
plt.xticks(bins, my_xticks)
plt.ylim(-20 * 10**(-8), 10 * 10**(-8))
plt.ylabel("Magnetic moment [Am^2]")
ax.errorbar(bins, means, yerr=stds, fmt='o')
'''
plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/NRM92_ARM_dispersion.png")

'''
point_names = df_names['sample']
for i, txt in enumerate(point_names):
    ax.annotate(txt, (NRM92_ARM_values[i], dispersion_values[i]))
#plt.xlim(0.02, 0.075)
plt.ylabel("S")
plt.xlabel("NRM92 / ARM")
plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/NRM92_graph_to_dispersion.png")


#print(NRM92_ARM_values)
for jt in range(len(df_names['sample'])):
    filepath = "..//..//ARM_final_data//" + df_names['sample'][jt] + "B.pmd"
    filename = os.path.relpath(filepath, cur_path)

    df = pd.read_csv(filename, sep='\s+', skiprows=3)
    df.columns = column_names

    Xc = df['Xc']
    Yc = df['Yc']
    Zc = df['Zc']

    I, D = [], []

    for i in range(len(Xc)):
        B = sqrt(Xc[i] ** 2 + Yc[i] ** 2 + Zc[i] ** 2)
        D.append(arctan(Yc[i] / Xc[i]))
        I.append(arcsin(Zc[i] / B))

    for i in range(len(I)):
        if D[i] < 0:
            D[i] = D[i] + 2 * np.pi

    for i in range(len(I)):
        if I[i] < 0:
            I[i] = I[i] * (-1)
            D[i] = D[i] + np.pi

    #print(np.degrees(I), np.degrees(D))
    # Create a stereonet with grid lines.
    fig, ax = mpls.subplots(figsize=(9, 6))
    ax.grid(color='k', alpha=0.2)
    ax.line(np.degrees(I), np.degrees(D), 'k^', markersize=10)
    ax.line(I_ref, D_ref, 'ro', markersize=10)
    plt.title(df_names["sample"][jt] + "  Degree of anisotropy: " + str(np.floor(df_names["aniso_percentage"][jt])) + "%")
    plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/PreMeanAnisoData/" + df_names["sample"][jt] + "B.png")
'''
