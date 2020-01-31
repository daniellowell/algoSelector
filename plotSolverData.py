


solversDict = { "ConvAsm3x3U": 1,
			"ConvAsm1x1U": 2,
			"ConvAsm1x1UV2": 3,
			"ConvBiasActivAsm1x1U": 4,
			"ConvAsm5x10u2v2f1": 5,
			"ConvAsm5x10u2v2b1": 6,
			"ConvAsm7x7c3h224w224k64u2v2p3q3f1": 7,
			"ConvOclDirectFwd11x11": 8,
			"ConvOclDirectFwdGen": 9,
			"ConvOclDirectFwd3x3": 10,
			"ConvOclDirectFwd": 11,
			"ConvOclDirectFwdFused": 12,
			"ConvOclDirectFwd1x1": 13,
			"ConvBinWinograd3x3U": 14,
			"ConvBinWinogradRxS": 15,
			"ConvAsmBwdWrW3x3": 16,
			"ConvAsmBwdWrW1x1": 17,
			"ConvOclBwdWrW2<1>": 18,
			"ConvOclBwdWrW2<2>": 19,
			"ConvOclBwdWrW2<4>": 20,
			"ConvOclBwdWrW2<8>": 21,
			"ConvOclBwdWrW2<16>": 22,
			"ConvOclBwdWrW2NonTunable": 23,
			"ConvOclBwdWrW53": 24,
			"ConvOclBwdWrW1x1": 25,
			"ConvHipImplicitGemmV4R1Fwd": 26,
			"ConvHipImplicitGemmV4Fwd": 27,
			"ConvHipImplicitGemmV4_1x1": 28,
			"ConvHipImplicitGemmV4R4FwdXdlops": 29,
			"ConvHipImplicitGemmV4R4Xdlops_1x1": 30,
			"ConvHipImplicitGemmV4R1WrW": 31,
			"ConvHipImplicitGemmV4WrW": 32,
			"gemm": 33,
			"fft": 34,
			"ConvWinograd3x3MultipassWrW<3, 4>": 35,
			"ConvBinWinogradRxSf3x2": 36,
			"ConvWinograd3x3MultipassWrW<3, 5>": 37,
			"ConvWinograd3x3MultipassWrW<3, 6>": 38,
			"ConvWinograd3x3MultipassWrW<3, 2>": 39,
			"ConvWinograd3x3MultipassWrW<3, 3>": 40,
			"ConvWinograd3x3MultipassWrW<7, 2>": 41,
			"ConvWinograd3x3MultipassWrW<7, 3>": 42,
			"ConvWinograd3x3MultipassWrW<7, 2, 1, 1>": 43,
			"ConvWinograd3x3MultipassWrW<7, 3, 1, 1>": 44,
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 2>": 45,
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 3>": 46,
			"ConvWinograd3x3MultipassWrW<5, 3>": 47,
			"ConvWinograd3x3MultipassWrW<5, 4>": 48,
			"ConvHipImplicitGemmV4R4WrWXdlops": 49,
			"ConvHipImplicitGemmV4R4GenFwdXdlops": 50,
			"ConvHipImplicitGemmV4R4GenWrWXdlops": 51,
			"ConvBinWinogradRxSf2x3": 52,
}


solverNames = [ "ConvAsm3x3U",
			"ConvAsm1x1U",
			"ConvAsm1x1UV2",
			"ConvBiasActivAsm1x1U",
			"ConvAsm5x10u2v2f1",
			"ConvAsm5x10u2v2b1",
			"ConvAsm7x7c3h224w224k64u2v2p3q3f1",
			"ConvOclDirectFwd11x11",
			"ConvOclDirectFwdGen",
			"ConvOclDirectFwd3x3",
			"ConvOclDirectFwd",
			"ConvOclDirectFwdFused",
			"ConvOclDirectFwd1x1",
			"ConvBinWinograd3x3U",
			"ConvBinWinogradRxS",
			"ConvAsmBwdWrW3x3",
			"ConvAsmBwdWrW1x1",
			"ConvOclBwdWrW2<1>",
			"ConvOclBwdWrW2<2>",
			"ConvOclBwdWrW2<4>",
			"ConvOclBwdWrW2<8>",
			"ConvOclBwdWrW2<16>",
			"ConvOclBwdWrW2NonTunable",
			"ConvOclBwdWrW53",
			"ConvOclBwdWrW1x1",
			"ConvHipImplicitGemmV4R1Fwd",
			"ConvHipImplicitGemmV4Fwd",
			"ConvHipImplicitGemmV4_1x1",
			"ConvHipImplicitGemmV4R4FwdXdlops",
			"ConvHipImplicitGemmV4R4Xdlops_1x1",
			"ConvHipImplicitGemmV4R1WrW",
			"ConvHipImplicitGemmV4WrW",
			"gemm",
			"fft",
			"ConvWinograd3x3MultipassWrW<3, 4>",
			"ConvBinWinogradRxSf3x2",
			"ConvWinograd3x3MultipassWrW<3, 5>",
			"ConvWinograd3x3MultipassWrW<3, 6>",
			"ConvWinograd3x3MultipassWrW<3, 2>",
			"ConvWinograd3x3MultipassWrW<3, 3>",
			"ConvWinograd3x3MultipassWrW<7, 2>",
			"ConvWinograd3x3MultipassWrW<7, 3>",
			"ConvWinograd3x3MultipassWrW<7, 2, 1, 1>",
			"ConvWinograd3x3MultipassWrW<7, 3, 1, 1>",
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 2>",
			"ConvWinograd3x3MultipassWrW<1, 1, 7, 3>",
			"ConvWinograd3x3MultipassWrW<5, 3>",
			"ConvWinograd3x3MultipassWrW<5, 4>",
			"ConvHipImplicitGemmV4R4WrWXdlops",
			"ConvHipImplicitGemmV4R4GenFwdXdlops",
			"ConvHipImplicitGemmV4R4GenWrWXdlops", 
			"ConvBinWinogradRxSf2x3"]

import sys
import subprocess
import csv
import numpy as np

strargs = sys.argv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

data = np.loadtxt(strargs[1], delimiter=",")

print(data[:10])
fig, axs = plt.subplots()
axs.hist(data, bins=len(solverNames), log=True)

fig.canvas.draw()

# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels[1] = 'Testing'

#plt.xticks(solverNames)
axs.set_xticks([i for i in range(len(solverNames))])
axs.set_xticklabels(solverNames)
for tick in axs.get_xticklabels():
    tick.set_rotation(90)

plt.show()



