import os
import numpy as np
import pandas as pd
from lib.scaling_laws import ScalingLawCalculator
from lib.probability_functions import normals_srl, randoms_srl, normal_m, density_to_prob
from matplotlib import pyplot as plt


## ======================================================================
# Define SRL exploration values
## ======================================================================

min_SRL = 5000
max_SRL = 100000
step = 500
SRL = np.arange(min_SRL, max_SRL+step, step)
SRL_sd = np.zeros(len(SRL))
kin = ["N", "R", "SS", "All","SCR"]
table = np.empty((len(kin), 2), dtype=object)

## ======================================================================
# Define scaling relations to use
## ======================================================================

methods = ["WellsAndCoppersmith94","Leonard2010", "Thingbaijam2017", "Brengman2019"]

## ==================================================================================
# Compute pool of magnitudes and standard deviations from the different scaling laws
## ==================================================================================

for k in range(len(kin)):
    kinematics = kin[k]
    array_Mw = []
    array_sd = []
    SRL_i = []
    for i in range(len(SRL)):
        scaling = ScalingLawCalculator(SRL[i])
        if kinematics not in ["SCR", "subduction", "All"]:
            mult = len(methods)
            results = [getattr(scaling, m)(kinematics) for m in methods]
            Mw_list, sd_list = zip(*results)
            array_Mw.append(list(Mw_list))
            array_sd.append(list(sd_list))
        elif kinematics in ["SCR"]:
            mult = 1
            Mw_list, sd_list = scaling.Leonard2010(kinematics)
            array_Mw.append([Mw_list])
            array_sd.append([sd_list])
        elif kinematics in ["subduction"]:
            mult = 1
            Mw_list, sd_list = scaling.Thingbaijam2017(kinematics)
            array_Mw.append([Mw_list])
            array_sd.append([sd_list])
        elif kinematics in ["All"]:
            methods = ["WellsAndCoppersmith94","Brengman2019"]
            mult = len(methods)
            results = [getattr(scaling, m)(kinematics) for m in methods]
            Mw_list, sd_list = zip(*results)
            array_Mw.append(list(Mw_list))
            array_sd.append(list(sd_list))
    Mw = np.array([float(i) for list in array_Mw for i in list])
    sd = np.array([float(i) for list in array_sd for i in list])

## ===============================================================================================
# Random sampling of magnitude values within the Mw+sd range for a better uncertainty exploration
## ===============================================================================================
    fig, ax = plt.subplots()
    samples = 1000
    matrix_M = np.empty((int(len(SRL)), samples*mult))
    M_range = np.linspace(4, 9, 1000)
    SRL_all = list(SRL)*mult
    SRL_all = [float(x) for x in SRL_all]
    percentiles = []
    pos = 0
    for m in range(len(Mw)):
        row = m//mult
        pos = (m % mult)*samples
        M_pdf = normal_m(Mw[m], sd[m], M_range)
        random_M = randoms_srl(M_range, M_pdf, samples)
        filter_samples = random_M
        filter_samples[(filter_samples < Mw[m] - sd[m]) | (filter_samples > Mw[m] + sd[m])] = np.nan
        matrix_M[row, pos: pos+ samples] = filter_samples
    for j in range(len(SRL)):
        perc = np.nanpercentile(matrix_M[j,:], 80)
        percentiles.append(perc)
        plt.scatter(np.zeros(len(matrix_M[j,:])) + SRL[j], matrix_M[j, :], s=.001, c="grey")
    percentiles = [float(x) for x in percentiles]
    plt.scatter(SRL, percentiles, c="red", s=4)
    loc_perc = np.where(np.array(percentiles)>=6.5)[0][0]
    plt.plot( [SRL[loc_perc], SRL[loc_perc]], [min(M_range), 6.5], c="black")
    plt.plot( [0, SRL[loc_perc]], [6.5,6.5], c="black")
    plt.text(SRL[loc_perc], 4.5, "  SRL threshold: "+str(round(SRL[loc_perc]/1e3))+"km", horizontalalignment="left")
    plt.xlim(min(SRL), max(SRL))
    plt.ylim(min(M_range), max(M_range))
    plt.ylabel("Mw")
    plt.xlabel("SRL (m)")
    plt.title("SRL threshold for "+kin[k] + " kinematics")
    fig.show()
    fig.savefig("Results/SRL_threshold_" + kin[k] + ".pdf")

    table[k,0] = kin[k]
    table[k,1] = SRL[loc_perc]

## ===============================================================================================
# Export files of the analysis
## ===============================================================================================

os.makedirs("Results", exist_ok=True)
df = pd.DataFrame(table, columns=["Kinematics", "SRL(m) - perc. 80%"])
pd.DataFrame(df).to_csv("Results/SRL_threshold.csv", index=False)