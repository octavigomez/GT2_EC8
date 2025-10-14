import os
import numpy as np
import pandas as pd
from lib.scaling_laws import ScalingLawCalculator
from lib.probability_functions import normals_srl, randoms_srl, normal_m, density_to_prob
from matplotlib import pyplot as plt

## ======================================================================
# Import SRL and kinematics from file
## ======================================================================

file = pd.read_csv("Faults_lengths.csv", delimiter=";")
SRL = np.array(file.Length)*1e3
SRL_sd = np.array(file.Unc)*1e3
kin = np.array(file.Kinematics) # N for normal, R for reverse, SS for strike-slip, SCR for stable continental region (just Leonard 2010) and subduction for subduction interfaces (just for Thinbaijam 2017)
Faults = np.array(file.Fault)
num_samples = 5000
table = np.empty((len(SRL), 6), dtype=object)

## ======================================================================
# Define scaling relations to use
## ======================================================================

methods = ["WellsAndCoppersmith94","Leonard2010", "Thingbaijam2017", "Brengman2019", "Blaser2010", "Wesnousky2008"] #,Ambraseys1998]

## ======================================================================
# Sample SRL values from the SRL probability distribution
## ======================================================================
for srl in range(len(SRL)):
    kinematics = kin[srl]
    SRL_sample = np.linspace(SRL[srl] - SRL_sd[srl] * 2, SRL[srl] + SRL_sd[srl] * 2, num_samples)
    normal_srl = normals_srl(SRL[srl], SRL_sd[srl], SRL_sample)
    random_srl = randoms_srl(SRL_sample, normal_srl, num_samples=num_samples)

    #Random sampling figure
    outputs_dir = os.makedirs("Results/"+Faults[srl], exist_ok=True)
    path_out = "Results/"+Faults[srl]
    # fig1, ax = plt.subplots()
    # plt.plot(SRL_sample, normal_srl, c="red", label="Normal SRL distribution")
    # plt.scatter(random_srl, np.zeros(len(random_srl)), marker="x", label="Random SRL samples")
    # plt.legend()
    # plt.xlabel("SRL (m)")
    # plt.ylabel("Probability density")
    # plt.title("Random sampling from SRL distribution \n Nsamples = " + str(num_samples) + "\n "+ Faults[srl])
    # plt.tight_layout()
    # plt.show()

    ## ==================================================================================
    # Compute pool of magnitudes and standard deviations from the different scaling laws
    ## ==================================================================================

    array_Mw = []
    array_sd = []
    for i in range(len(random_srl)):
        scaling = ScalingLawCalculator(random_srl[i])
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
            methods = ["WellsAndCoppersmith94","Brengman2019", "Blaser2010", "Wesnousky2008"] #,"Ambraseys1998"]
            mult = len(methods)
            results = [getattr(scaling, m)(kinematics) for m in methods]
            Mw_list, sd_list = zip(*results)
            array_Mw.append(list(Mw_list))
            array_sd.append(list(sd_list))
    Mw = np.array([float(i) for list in array_Mw for i in list])
    sd = np.array([float(i) for list in array_sd for i in list])

    ## ======================================================================
    # Compute PDFs for each magnitude estimation and store them in a matrix
    ## ======================================================================

    M_range = np.linspace(4, 9, 1000)
    array_PDFs = np.zeros((len(Mw), len(M_range)))

    for j in range(len(Mw)):
        M_pdf = normal_m(Mw[j], sd[j], M_range)
        dx = float(np.diff(M_range)[0])
        M_probabilities = density_to_prob(M_range, M_pdf)
        array_PDFs[j, :] = M_probabilities
    mean_pdf = np.mean(array_PDFs, axis=0)
    p = density_to_prob(M_range, mean_pdf)
    mean_stat = np.sum(p*M_range)
    sd_stat = np.sqrt(np.sum(p*(M_range-mean_stat)**2))


    #Plot results
    fig2, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for i in range(len(Mw)):
        ax1.plot(M_range, array_PDFs[i,:], color="grey", linestyle="dashed", linewidth=0.1)
    ax2.plot(M_range, mean_pdf, color="red", label="Average PDF")
    ax1.plot([6.5,6.5], [0, array_PDFs.max()+0.001], color="black", linestyle=":")
    ax2.scatter(mean_stat, max(mean_pdf), c="red")
    ax2.plot([mean_stat-sd_stat, mean_stat+sd_stat], [max(mean_pdf),max(mean_pdf)], c="red")
    ax1.set_xlabel("Mw")
    ax1.set_ylabel("Probability")
    ax2.set_ylabel("Probability", color="red")
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax1.set_ylim(0, array_PDFs.max()+0.001)
    ax2.set_ylim(0, max(mean_pdf+0.001))
    ax1.set_xlim(4, 8.5)
    plt.title("Magnitude PDF \n Mean = " + str(round(mean_stat,2))+"±"+str(round(sd_stat,2))+ "\n "+ Faults[srl])
    plt.legend()
    plt.tight_layout()
    plt.show()


    ## ======================================================================
    # Compute CDF for magnitude distribution
    ## ======================================================================

    M_cdf = np.cumsum(p)
    threshold = 0.8 # 15% of the distribution
    thr = np.interp(threshold, M_cdf, M_range)

    #Plot
    fig3, ax = plt.subplots()
    thr_pdf = thr/np.sum(thr*dx)
    plt.plot(M_range, mean_pdf/np.max(mean_pdf), c="black", label="Average PDF")
    plt.plot(M_range, M_cdf/np.max(M_cdf), c="red", label = "CDF")
    plt.plot([thr,thr], [0, 1], c="blue", linestyle="dashed", label = str(threshold*100)+ "% threshold")
    plt.text(thr, 1.02, str(round(thr,2)), ha="center", va="bottom")
    plt.ylim(0, 1.1)
    plt.xlim(5.5, 8.5)
    plt.xlabel("Mw")
    plt.ylabel("Normalized probability")
    plt.title("Magnitude quantile analysis"+ "\n "+ Faults[srl])
    plt.legend()
    plt.tight_layout()
    plt.show()

    ## ======================================================================
    # Export files
    ## ======================================================================

    table[srl,0] = Faults[srl]
    table[srl,1] = SRL[srl]/1e3
    table[srl,2] = SRL_sd[srl]/1e3
    table[srl,3] = mean_stat
    table[srl,4] = sd_stat
    table[srl,5] = thr

    #fig1.savefig(path_out + "/Random_sampling.png")
    fig2.savefig(path_out + "/Magnitude_PDF.png")
    fig3.savefig(path_out+ "/Quantile_analysis.png")

table = pd.DataFrame(table, columns=["Fault", "Length", "Length_unc", "Mw", "Mw_sigma", str(threshold*100)+ "% quantile"])
table.to_csv("Results/final_stats.csv")