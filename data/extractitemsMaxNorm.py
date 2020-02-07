import sys
import subprocess
import csv

strargs = sys.argv

solvers = { "ConvAsm3x3U": 1,
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



lhs=[]
rhs=[]
with open(strargs[1], 'r') as fp:
	for line in enumerate(fp):
		twoparts = line[1].split('=')
		lhs.append(twoparts[0])		
		rhs.append(twoparts[1])

csvrow=[]
dirval={"F": 0.0, "B": 0.5, "W": 1.0,}  
C=0.0
Hin=0.0
Win=0.0
n=0.0
K=0.0
x=0.0
y=0.0
Hout=0.0
Wout=0.0
padH=0.0
padW=0.0
strideH=0.0
strideW=0.0
dilationH=0.0 
dilationW=0.0
group=0.0
for i in range(len(lhs)):
	rowsplit=lhs[i].split('-')
	csvrow.append([])
# K, Hout, Wout, n, padH, padW, strideH, strideW, dilationH, dilationW, bias, layout, type, direction
	csvrow[i].append(float(rowsplit[0])) # C
	C=max(C,float(rowsplit[0]))
	csvrow[i].append(float(rowsplit[1])) # Hin
	Hin=max(Hin,float(rowsplit[1]))
	csvrow[i].append(float(rowsplit[2])) # Win
	Win=max(Win,float(rowsplit[2]))
	csvrow[i].append(float(rowsplit[3].split('x')[0])) # x
	x=max(x,float(rowsplit[3].split('x')[0]))
	csvrow[i].append(float(rowsplit[3].split('x')[1])) # y
	y=max(y,float(rowsplit[3].split('x')[0]))
	csvrow[i].append(float(rowsplit[4])) # K
	K=max(K,float(rowsplit[4]))
	csvrow[i].append(float(rowsplit[5])) # Hout
	Hout=max(Hout,float(rowsplit[5]))
	csvrow[i].append(float(rowsplit[6])) # Wout
	Wout=max(Wout,float(rowsplit[6]))
	csvrow[i].append(float(rowsplit[7])) # n
	n=max(n,float(rowsplit[7]))
	csvrow[i].append(float(rowsplit[8].split('x')[0])+1) # pad H
	padH=max(padH,float(rowsplit[8].split('x')[0])+1)
	csvrow[i].append(float(rowsplit[8].split('x')[1])+1) # pad W
	padW=max(padW,float(rowsplit[8].split('x')[1])+1)
	csvrow[i].append(float(rowsplit[9].split('x')[0])) # stride H
	strideH=max(strideH,float(rowsplit[9].split('x')[0]))
	csvrow[i].append(float(rowsplit[9].split('x')[1])) # stride W
	strideW=max(strideW,float(rowsplit[9].split('x')[1]))
	csvrow[i].append(float(rowsplit[10].split('x')[0])) # dilation H
	dilationH=max(dilationH,float(rowsplit[10].split('x')[0]))
	csvrow[i].append(float(rowsplit[10].split('x')[1])) # dilation W
	dilationW=max(dilationW,float(rowsplit[10].split('x')[1]))
	csvrow[i].append(dirval[rowsplit[14].split('_')[0]]) # direction
	if len(rowsplit[14].split('_')) > 1:
		csvrow[i].append(float(rowsplit[14].split('_')[1].strip('g'))) # group
		group=max(group,float(rowsplit[14].split('_')[1].strip('g')))
	else:
		csvrow[i].append(float(0.00001)) # group


for i in range(len(csvrow)):
	csvrow[i][0]=float(csvrow[i][0]/C)
	csvrow[i][1]=float(csvrow[i][1]/Hin)
	csvrow[i][2]=float(csvrow[i][2]/Win)
	csvrow[i][3]=float(csvrow[i][3]/x)
	csvrow[i][4]=float(csvrow[i][4]/y)
	csvrow[i][5]=float(csvrow[i][5]/K)
	csvrow[i][6]=float(csvrow[i][6]/Hout)
	csvrow[i][7]=float(csvrow[i][7]/Wout)
	csvrow[i][8]=float(csvrow[i][8]/n)
	csvrow[i][9]=float(csvrow[i][9]/padH)
	csvrow[i][10]=float(csvrow[i][10]/padW)
	csvrow[i][11]=float(csvrow[i][11]/strideH)
	csvrow[i][12]=float(csvrow[i][12]/strideW)
	csvrow[i][13]=float(csvrow[i][13]/dilationH)
	csvrow[i][14]=float(csvrow[i][14]/dilationW)
	csvrow[i][16]=float(csvrow[i][16]/group)


with open(strargs[2], 'w') as infile:
	config_writer = csv.writer(infile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(lhs)):
		config_writer.writerow(csvrow[i])

# START LABELS PARSING
# here we would like to extract the smallest time.

csvlblrow=[]
for row in rhs:
	rowsplit=row.split(';')
	s=[]
	for entry in rowsplit:
		try:
			elems=entry.split(':')[1].split(',')
		except:
			print("row: "+row+", entry: "+entry)
			exit()
		s.append([elems[0],float(elems[1])])
	csvlblrow.append(sorted(s, key=lambda s: s[1])[0][0])
	
	
with open(strargs[3], 'w') as lblfile:
        config_writer = csv.writer(lblfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(rhs)):
                config_writer.writerow([solvers[csvlblrow[i]]-1])

with open('names_'+strargs[3], 'w') as nmfile:
	config_writer = csv.writer(nmfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(rhs)):
		config_writer.writerow([csvlblrow[i]])



