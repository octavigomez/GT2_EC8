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
max_SRL = 30000
step = 250
SRL = np.arange(min_SRL, max_SRL+step, step)
SRL_sd = np.zeros(len(SRL))
kin = ["N", "R", "SS", "All"]
type = ["just_SRL", "All"]
table = np.empty((len(kin), 3), dtype=object)
os.makedirs("Results", exist_ok=True)

## ======================================================================
# Define scaling relations to use
## ======================================================================

methods = ["WellsAndCoppersmith94","Leonard2010", "Thingbaijam2017", "Brengman2019", "Blaser2010", "Wesnousky2008"] #,Ambraseys1998]
methods_srl = ["WellsAndCoppersmith94","Leonard2010", "Wesnousky2008"]
methods_all_type = ["WellsAndCoppersmith94","Brengman2019", "Blaser2010", "Wesnousky2008"]
methods_all_type_srl = ["WellsAndCoppersmith94", "Wesnousky2008"]

methods_full = [methods_srl, methods]
methods_all = [methods_all_type_srl, methods_all_type]
color = ["r", "green"]


## ==================================================================================
# Compute pool of magnitudes and standard deviations from the different scaling laws
## ==================================================================================

for k in range(len(kin)):
    kinematics = kin[k]
    fig, ax = plt.subplots()
    for mt in range(len(methods_full)):
        array_Mw = []
        array_sd = []
        SRL_i = []
        for i in range(len(SRL)):
            scaling = ScalingLawCalculator(SRL[i])
            if kinematics not in ["SCR", "subduction", "All"]:
                mult = len(methods_full[mt])
                results = [getattr(scaling, z)(kinematics) for z in methods_full[mt]]
                Mw_list, sd_list = zip(*results)
                array_Mw.append(list(Mw_list))
                array_sd.append(list(sd_list))
            elif kinematics in ["All"]:
                mult = len(methods_all[mt])
                results = [getattr(scaling, z)(kinematics) for z in methods_all[mt]]
                Mw_list, sd_list = zip(*results)
                array_Mw.append(list(Mw_list))
                array_sd.append(list(sd_list))
        Mw = np.array([float(i) for list in array_Mw for i in list])
        sd = np.array([float(i) for list in array_sd for i in list])

    ## ===============================================================================================
    # Random sampling of magnitude values within the Mw+sd range for a better uncertainty exploration
    ## ===============================================================================================
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
            filter_samples[(filter_samples < Mw[m] - sd[m]*3) | (filter_samples > Mw[m] + sd[m]*3)] = np.nan
            matrix_M[row, pos: pos+ samples] = filter_samples
        #     plt.plot(M_range, M_pdf, color="red")
        #     # plt.scatter(filter_samples, filter_samples*0, c="blue", marker="x", s= 2, label="Samples")
        #     plt.xlabel("Mw")
        #     plt.ylabel("Probability")
        #     plt.xlim(4, 8.5)
        #     # plt.legend()
        # plt.show()
        for j in range(len(SRL)):
            quantile = 80
            perc = np.nanpercentile(matrix_M[j,:], quantile)
            percentiles.append(perc)
            plt.scatter(np.zeros(len(matrix_M[j,:])) + SRL[j], matrix_M[j, :], s=.001, c=color[mt], alpha = 0.05)
            #plt.boxplot(matrix_M[j, ~np.isnan(matrix_M[j, :])], positions=[float(SRL[j])],
                        #widths=500, showfliers=False)
        percentiles = [float(x) for x in percentiles]
        loc_perc = np.where(np.array(percentiles) >= 6.5)[0][0]
        plt.scatter(SRL, percentiles, c=color[mt], s=8, label = "Q"+str(quantile)+type[mt])
        plt.plot( [SRL[loc_perc], SRL[loc_perc]], [min(M_range), 6.5], c=color[mt])
        plt.plot( [0, SRL[loc_perc]], [6.5,6.5], c=color[mt])
        plt.text(SRL[loc_perc], 4.5, " " +str(round(SRL[loc_perc]/1e3, 1))+"km",
                 horizontalalignment="left", color=color[mt], rotation=90)
        plt.ylim(min(M_range), max(M_range))
        plt.xticks(ticks = np.arange(min(SRL), max(SRL)+10000, 10000))
        ax.set_xticklabels(f"{x/1000:.0f}" for x in np.arange(min(SRL), max(SRL)+10000, 10000))
        plt.xlim(min(SRL), max(SRL))
        plt.ylabel("Mw")
        plt.xlabel("SRL (km)")
        plt.title("SRL threshold for type "+kin[k])
        plt.grid("on", which="major", linestyle=":")
        plt.legend()

        table[k,0] = kin[k]
        if mt ==0:
            cl=1
        elif mt ==1:
            cl=2
        table[k, cl] = SRL[loc_perc]

    fig.show()
    fig.savefig("Results/SRL_threshold_" + kin[k] + ".pdf")

## ===============================================================================================
# Export files of the analysis
## ===============================================================================================

df = pd.DataFrame(table, columns=["Kinematics", "Genuine SRL(m) - perc. 80%", "All SRL(m) - perc. 80%"])
pd.DataFrame(df).to_csv("Results/SRL_threshold.csv", index=False)



