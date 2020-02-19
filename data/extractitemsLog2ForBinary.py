import sys
import subprocess
import csv
import numpy as np

strargs = sys.argv

lhs=[]
rhs=[]
with open(strargs[1], 'r') as fp:
	for line in enumerate(fp):
		twoparts = line[1].split('=')
		lhs.append(twoparts[0])		
		rhs.append(twoparts[1])

# START DATA PARSING
csvrow=[]
dirval={"F": 1.0, "B": 2.0, "W": 3.0,}  
direction=[]
for i in range(len(lhs)):
	rowsplit=lhs[i].split('-')
	csvrow.append([])
	csvrow[i].append(np.log2(float(rowsplit[0]))) # C
	csvrow[i].append(np.log2(float(rowsplit[1]))) # Hin
	csvrow[i].append(np.log2(float(rowsplit[2]))) # Win
	csvrow[i].append(np.log2(float(rowsplit[3].split('x')[0]))) # x
	csvrow[i].append(np.log2(float(rowsplit[3].split('x')[1]))) # y
	csvrow[i].append(np.log2(float(rowsplit[4]))) # K
	csvrow[i].append(np.log2(float(rowsplit[5]))) # Hout
	csvrow[i].append(np.log2(float(rowsplit[6]))) # Wout
	csvrow[i].append(np.log2(float(rowsplit[7]))) # n
	csvrow[i].append(np.log2(float(rowsplit[8].split('x')[0]) + 1)) # pad H
	csvrow[i].append(np.log2(float(rowsplit[8].split('x')[1]) + 1)) # pad W
	csvrow[i].append(np.log2(float(rowsplit[9].split('x')[0]))) # stride H
	csvrow[i].append(np.log2(float(rowsplit[9].split('x')[1]))) # stride W
	csvrow[i].append(np.log2(float(rowsplit[10].split('x')[0]))) # dilation H
	csvrow[i].append(np.log2(float(rowsplit[10].split('x')[1]))) # dilation W
	direction.append(rowsplit[14].split('_')[0])
	csvrow[i].append(float(dirval[rowsplit[14].split('_')[0]])) # direction
	if len(rowsplit[14].split('_')) > 1:
		csvrow[i].append(np.log2(float(rowsplit[14].split('_')[1].strip('g')))) # group
	else:
		csvrow[i].append(float(0.00001)) # group


with open(strargs[2], 'w') as infile:
	config_writer = csv.writer(infile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(lhs)):
		config_writer.writerow(csvrow[i])

# DONE DATA

# START LABELS PARSING
csvlblrow=[]
for row in rhs:
	rowsplit=row.split(';')
	s=[]
	for entry in rowsplit:
		elems=entry.split(':')[1].split(',')
		s.append([elems[0],float(elems[1])])
	csvlblrow.append(sorted(s, key=lambda s: s[1])[0][0])
	
	
with open(strargs[3], 'w') as lblfile:
	config_writer = csv.writer(lblfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for i in range(len(rhs)):
		solution=csvlblrow[i]
		if solution != "gemm":
			config_writer.writerow([0])
		else:
			config_writer.writerow([1])


