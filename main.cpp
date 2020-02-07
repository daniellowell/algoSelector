#include <fdeep/fdeep.hpp>

#include <iostream>
#include <cmath>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


// USAGE: ./main  <json file path>

int main(int argc, char *argv[]){

	std::vector<std::string> solvers = { 
			"ConvAsm3x3U",
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
			"ConvBinWinogradRxSf2x3"
	};


	std::vector<std::string> feature_names= { "C","Hin", "Win", "x", "y", "K", "Hout", "Wout", "n", "padH", "padW", "strideH", "strideW", "dilationH", "dilationW", "group" };
	const auto model = fdeep::load_model(argv[1]);

	// Sample input:
	const std::vector<float> input = {1.0,3.0,3.0,0.0,0.0,1.0,3.321928094887362,3.321928094887362,1.584962500721156,1.0,1.0,0.0,0.0,0.0,0.0,0.0001};
	const fdeep::shared_float_vec sinput(fplus::make_shared_ref<fdeep::float_vec>(std::move(input)));

	const auto result = model.predict_class({
	        fdeep::tensor5(fdeep::shape5(1, 1, 1, feature_names.size(), 1), sinput)});

	std::cout << "Raw result: " << result << std::endl;
	std::cout << "Solver classification: " << solvers[result - 1] << std::endl;

	return 0;
}




