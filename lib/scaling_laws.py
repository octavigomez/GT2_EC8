import numpy as np

class ScalingLawCalculator:
    def __init__(self, SRL):
        self.SRL = SRL

    def WellsAndCoppersmith94(self, kinematics):
        if kinematics == "N":
            a, b, sd = 4.86, 1.32, 0.34
        elif kinematics == "R":
            a, b, sd = 5.0, 1.22, 0.28
        elif kinematics == "SS":
            a, b, sd = 5.16, 1.12, 0.28
        elif kinematics == "All":
            a, b, sd = 5.08, 1.16, 0.28
        M = a + b * np.log10(self.SRL/1e3)
        return M, sd

    def Leonard2010(self, kinematics):
        if (kinematics == "N") | (kinematics == "R"):
            a, b, sd_b = 2.5, 7.96, [7.53, 8.51]
        elif kinematics == "SS":
            a, b, sd_b = 2.5, 7.85, [7.41, 8.28]
        elif kinematics == "SCR":
            a, b, sd_b = 2.5, 8.08, [7.87, 8.28]
        RLD = 10 ** ((np.log10(self.SRL) + 0.275) / 1.1)
        logM0 = a * np.log10(RLD) + b
        M = 2/3*logM0-6.07
        sd_b_fixed = (sd_b[1]-sd_b[0])/2
        sd = (2/3)*sd_b_fixed
        return M, sd

    def Thingbaijam2017(self, kinematics):
        if kinematics == "N":
            a, b, sd_a, sd_b, sd_logL = -1.722, 0.485, 0.260, 0.036, 0.128
        elif kinematics == "R":
            a, b, sd_a, sd_b, sd_logL = -2.693, 0.614, 0.292, 0.043, 0.083
        elif kinematics == "SS":
            a, b, sd_a, sd_b, sd_logL = -2.943, 0.681, 0.357, 0.052, 0.151
        elif kinematics == "subduction":
            a, b, sd_a, sd_b, sd_logL = -2.412, 0.583, 0.288, 0.037, 0.107
        M = (np.log10(self.SRL/1e3)-a)/b
        sd = sd_logL/b
        return M, sd

    def Brengman2019(self, kinematics):
        if kinematics == "N":
            a, b, sd_a, sd_b = 3.9568, 1.7917, 0.6761, 0.5074
        elif kinematics == "R":
            a, b, sd_a, sd_b = 4.2067, 1.7219, 0.3281, 0.1833
        elif kinematics == "SS":
            a, b, sd_a, sd_b = 4.8263, 1.2874, 0.5101, 0.3351
        elif kinematics == "All":
            a, b, sd_a, sd_b = 4.2089, 1.9771, 0.2873, 0.2058
        M = a + b * np.log10(self.SRL/1e3)
        sd = np.sqrt((sd_a)**2 + (np.log10(self.SRL/1e3)*sd_b)**2)
        return M, sd

    # def Ambraseys1998(self, kinematics):
    #     a, b = 4.9, 1.33
    #     M = a+b*np.log10(self.SRL/1e3)
    #     sd = 0
    #     return M, sd

    def Blaser2010(self, kinematics):
        if kinematics == "N":
            a, b, sd_a, sd_b, sdlogL = -1.91, 0.53, 0.29, 0.04, 0.18
        elif kinematics == "R":
            a, b, sd_a, sd_b, sdlogL = -2.37, 0.57, 0.13, 0.02, 0.18
        elif kinematics == "SS":
            a, b, sd_a, sd_b, sdlogL = -2.69, 0.64, 0.11, 0.02, 0.18
        elif kinematics == "All":
            a, b, sd_a, sd_b, sdlogL = -2.31, 0.57, 0.08, 0.01, 0.20
        SRL = self.SRL/0.75 #equation by Wells&Coppermsith94
        logL = np.log10(SRL/1e3)
        M = (logL-a)/b
        sd = sdlogL/b
        return M, sd

    def Wesnousky2008(self, kinematics):
        if kinematics == "N":
            a, b, sd = 6.12, 0.47, 0.27
        elif kinematics == "R":
            a, b, sd = 4.11, 1.88, 0.24
        elif kinematics == "SS":
            a, b, sd = 5.56, 0.87, 0.24
        elif kinematics == "All":
            a, b, sd = 5.3, 1.02, 0.28
        M = a + b * np.log10(self.SRL/1e3)
        return M, sd

    ## !!! I think the next regression is not orthogonal !!!

    # def MaiAndBeroza2000(self, kinematics): # Considering effective dimensions
    #     if (kinematics == "N") | (kinematics == "R"):
    #         a, b, sdlogL = -6.39, 0.4, 0.19
    #     elif kinematics == "SS":
    #         a, b, sdlogL = -6.31, 0.4, 0.12
    #     elif kinematics == "All":
    #         a, b, sdlogL = -6.13, 0.39, 0.16
    #     logL = np.log10(self.SRL/1e3)
    #     logM0 = (logL-a)/b
    #     M = 0.67*logM0-10.7
    #     sd = 0.67*sdlogL
    #     return M, sd7