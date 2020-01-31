#!/bin/bash

set -e

# cat gfx906_64.OpenCL.2_3_0.ufdb.txt gfx906_64.OpenCL.fdb.txt | grep FP32 > bigfileFP32_FBW.csv

cat ./bigfileFP32_FBW.txt | grep "\-FP32-F" > bigfileFP32_F.csv 
cat ./bigfileFP32_FBW.txt | grep "\-FP32-B" > bigfileFP32_B.csv 
cat ./bigfileFP32_FBW.txt | grep "\-FP32-W" > bigfileFP32_W.csv 

python3 extractitemsLog2Norm.py bigfileFP32_FBW.csv inlog2big_FBW.csv lbllog2big_FBW.csv
python3 extractitemsLog2Norm.py bigfileFP32_F.csv inlog2big_F.csv lbllog2big_F.csv
python3 extractitemsLog2Norm.py bigfileFP32_B.csv inlog2big_B.csv lbllog2big_B.csv
python3 extractitemsLog2Norm.py bigfileFP32_W.csv inlog2big_W.csv lbllog2big_W.csv

# python3 extractitemsMaxNorm.py bigfileFP32_FBW.txt inmaxnormbig_FBW.csv lblmaxnormbig_FBW.csv
# python3 extractitemsMaxNorm.py bigfileFP32_F.txt inmaxnormbig_F.csv lblmaxnormbig_F.csv
# python3 extractitemsMaxNorm.py bigfileFP32_B.txt inmaxnormbig_B.csv lblmaxnormbig_B.csv
# python3 extractitemsMaxNorm.py bigfileFP32_W.txt inmaxnormbig_W.csv lblmaxnormbig_W.csv

