#include <iostream>
#include "hls_stream.h"


#define CONV_IN_HEIGHT    224
#define CONV_IN_WIDTH     224
#define CONV_IN_CHANNELS_1 2 // same as Tn
#define CONV_OUT_CHANNELS_COM 1
#define KERNEL_SIZE       3
#define CONV_STRIDE       1
#define KERNEL_CHANNELS_COM   1     // same as the no. of CONV_OUT_CHANNELS_COM
#define padding           1     // user defined, not an input argument

// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)  // 6
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)    // 6

#define K KERNEL_SIZE


void parallel_1_conv2d(
	float IFM[CONV_IN_CHANNELS_1][CONV_IN_HEIGHT][CONV_IN_WIDTH],
	float F[KERNEL_CHANNELS_COM][CONV_IN_CHANNELS_1][KERNEL_SIZE][KERNEL_SIZE],
	float OFM[CONV_OUT_CHANNELS_COM][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
){
	// Array partitioning for input feature maps - complete partition for parallel access


	// Array partitioning for input feature maps - complete partition for parallel access







				    #pragma HLS ARRAY_PARTITION variable=IFM dim=1
				    #pragma HLS ARRAY_PARTITION variable=IFM dim=2
				    #pragma HLS ARRAY_PARTITION variable=IFM dim=3

				    // Array partitioning for filters - complete partition for kernel dimensions
				    #pragma HLS ARRAY_PARTITION variable=F dim=1
				    #pragma HLS ARRAY_PARTITION variable=F dim=2
				    #pragma HLS ARRAY_PARTITION variable=F dim=3
				    #pragma HLS ARRAY_PARTITION variable=F dim=4

				    // Array partitioning for output feature maps
				    #pragma HLS ARRAY_PARTITION variable=OFM dim=1
				    #pragma HLS ARRAY_PARTITION variable=OFM dim=2
				    #pragma HLS ARRAY_PARTITION variable=OFM dim=3








	// Initialize output feature map to zero
	init_output:
	for (int h = 0; h < CONV_OUT_HEIGHT; h++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_TRIPCOUNT min=8 max=8
		for (int w = 0; w < CONV_OUT_WIDTH; w++) {
			#pragma HLS UNROLL factor=8
			for (int oc = 0; oc < CONV_OUT_CHANNELS_COM; oc++) {
				#pragma HLS UNROLL factor=1
				OFM[oc][h][w] = 0;
			}
		}
	}

	ring_conv:
	for(int H=KERNEL_SIZE; H<CONV_IN_HEIGHT+2*padding; H=H+4) {
#pragma HLS PIPELINE II=1 rewind

		for(int W=KERNEL_SIZE-1; W<CONV_IN_WIDTH+2*padding; W=W+CONV_STRIDE) {

#pragma HLS UNROLL factor=15

			for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1) {
#pragma HLS UNROLL factor=1

				// Declare local accumulators for all OFM positions
				float local_ofm_0 = 0;  // For [H-K][W-K+1]
				float local_ofm_1 = 0;  // For [H-K+1][W-K+1]
				float local_ofm_2 = 0;  // For [H-K+2][W-K+1]
				float local_ofm_3 = 0;  // For [H-K+3][W-K+1]



#pragma HLS bind_op variable=local_ofm_0 op=fadd impl=dsp
#pragma HLS bind_op variable=local_ofm_1 op=fadd impl=dsp
#pragma HLS bind_op variable=local_ofm_2 op=fadd impl=dsp
#pragma HLS bind_op variable=local_ofm_3 op=fadd impl=dsp



				// Loop for multiple channels of IFM and single 3D kernel
				for(int ks=0; ks<CONV_IN_CHANNELS_1; ks++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=1
					//#pragma HLS LOOP_TRIPCOUNT min=4 max=4

					// Main conditions for right shifting
					if(W==K-1 && (H!=K)) { // case-1 start
						for(int n=0; n<K-1; n++) {
#pragma HLS UNROLL factor=2

							for(int m=0; m<2; m++) {
#pragma HLS UNROLL factor=2

								bool cond1 = !((H-1+m) <= padding-1 || (H-1+m) >= padding+CONV_IN_HEIGHT);
								bool cond2 = !((n+1) <= padding-1 || (n+1) >= padding+CONV_IN_WIDTH);
								bool combined_cond = cond1 && cond2;

								if(combined_cond) {
									float temp_mult1 = IFM[ks][H-1+m-padding][n+1-padding]*F[C][ks][m+K-2][n+1];
									float temp_mult2 = IFM[ks][H-1+m-padding][n+1-padding]*F[C][ks][m+K-1][n+1];



#pragma HLS bind_op variable=temp_mult1 op=fmul impl=dsp
#pragma HLS bind_op variable=temp_mult2 op=fmul impl=dsp



									local_ofm_1 += temp_mult1;
									if(m==0) {
										local_ofm_0 += temp_mult2;
									}
								}
							}
						}
					} // case-1 end
					else { // case-2 start
						for(int j=0; j<K; j++) {
#pragma HLS UNROLL factor=3

							for(int i=0; i<K; i++) {
#pragma HLS UNROLL factor=3

								// Precompute boolean expressions
								bool width_bounds_valid = !((W-K+1+j) <= padding-1 || (W-K+1+j) >= CONV_IN_WIDTH+padding);
								bool height_pos1_valid = !((i+H-K) <= padding-1 || (i+H-K) >= CONV_IN_HEIGHT+padding);
								bool height_pos2_valid = !((i+H-K+1) <= padding-1 || (i+H-K+1) >= CONV_IN_HEIGHT+padding);
								bool height_pos3_valid = !((i+H-K+2) <= padding-1 || (i+H-K+2) >= CONV_IN_HEIGHT+padding);
								bool height_pos4_valid = !((i+H-K+3) <= padding-1 || (i+H-K+3) >= CONV_IN_HEIGHT+padding);

								// Additional condition checks
								bool is_width_boundary = (W == CONV_IN_WIDTH + 2*padding - 1);
								bool is_not_height_boundary = (H != CONV_IN_HEIGHT + 2*padding - 1);
								bool edge_case_condition = is_width_boundary && is_not_height_boundary;
								bool filter_pos_valid_k2 = (i <= K-2);
								bool filter_pos_valid_k3 = (i <= K-3);

								if(width_bounds_valid) {
									float mult_result_0 = IFM[ks][i+H-K-padding][W-K+1+j-padding] * F[C][ks][i][j];
									float mult_result_1 = IFM[ks][i+H-K+1-padding][W-K+1+j-padding] * F[C][ks][i][j];
									float mult_result_2 = IFM[ks][i+H-K+2-padding][W-K+1+j-padding] * F[C][ks][i][j];
									float mult_result_3 = IFM[ks][i+H-K+3-padding][W-K+1+j-padding] * F[C][ks][i][j];


#pragma HLS bind_op variable=mult_result_0 op=fmul impl=dsp
#pragma HLS bind_op variable=mult_result_1 op=fmul impl=dsp
#pragma HLS bind_op variable=mult_result_2 op=fmul impl=dsp
#pragma HLS bind_op variable=mult_result_3 op=fmul impl=dsp



									if(height_pos1_valid) {
										local_ofm_0 += mult_result_0;
									}

									if(height_pos2_valid) {
										local_ofm_1 += mult_result_1;
									}

									if(edge_case_condition) {
										if(filter_pos_valid_k2) {
											if(height_pos3_valid) {
												local_ofm_2 += mult_result_2;
											}

											if(filter_pos_valid_k3) {
												if(height_pos4_valid) {
													local_ofm_3 += mult_result_3;
												}
											}
										}
									}
								}
							}
						}
					} // case-2 end
				} // ks loop

				// Write all accumulated values to OFM at once
				float temp_add_0 = OFM[C][H-K][W-K+1] + local_ofm_0;
				float temp_add_1 = OFM[C][H-K+1][W-K+1] + local_ofm_1;
				float temp_add_2 = OFM[C][H-K+2][W-K+1] + local_ofm_2;
				float temp_add_3 = OFM[C][H-K+3][W-K+1] + local_ofm_3;



#pragma HLS bind_op variable=temp_add_0 op=fadd impl=dsp
#pragma HLS bind_op variable=temp_add_1 op=fadd impl=dsp
#pragma HLS bind_op variable=temp_add_2 op=fadd impl=dsp
#pragma HLS bind_op variable=temp_add_3 op=fadd impl=dsp



				OFM[C][H-K][W-K+1] = temp_add_0;
				OFM[C][H-K+1][W-K+1] = temp_add_1;

				bool is_width_boundary = (W == CONV_IN_WIDTH + 2*padding - 1);
				bool is_not_height_boundary = (H != CONV_IN_HEIGHT + 2*padding - 1);
				if(is_width_boundary && is_not_height_boundary) {
					OFM[C][H-K+2][W-K+1] = temp_add_2;
					OFM[C][H-K+3][W-K+1] = temp_add_3;
				}
			} // C loop
		} // W loop
	} // H loop-1 ends

	// Second H loop for reverse direction processing
	for(int H=KERNEL_SIZE+2; H<CONV_IN_HEIGHT+2*padding; H=H+4) {
#pragma HLS PIPELINE II=1 rewind
//#pragma HLS LOOP_TRIPCOUNT min=2 max=3
//#pragma HLS UNROLL factor=15
		for(int W=CONV_IN_WIDTH-KERNEL_SIZE+padding+1; W>=0; W=W-CONV_STRIDE) {
//#pragma HLS LOOP_TRIPCOUNT min=8 max=9
#pragma HLS UNROLL factor=15

			for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1) {
#pragma HLS UNROLL factor=1

				// Local accumulators for this version
				float local_2_ofm_0 = 0;  // For [H-K][W]
				float local_2_ofm_1 = 0;  // For [H-K+1][W]
				float local_2_ofm_2 = 0;  // For [H-K+2][W]
				float local_2_ofm_3 = 0;  // For [H-K+3][W]



#pragma HLS bind_op variable=local_2_ofm_0 op=fadd impl=dsp
#pragma HLS bind_op variable=local_2_ofm_1 op=fadd impl=dsp
#pragma HLS bind_op variable=local_2_ofm_2 op=fadd impl=dsp
#pragma HLS bind_op variable=local_2_ofm_3 op=fadd impl=dsp



				for(int ks=0; ks<CONV_IN_CHANNELS_1; ks++) {
//#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=1
// #pragma HLS LOOP_TRIPCOUNT min=4 max=4

					if((W==CONV_IN_WIDTH+2*padding-K)) { // case-1 start
						for(int n=0; n<K-1; n++) {
#pragma HLS UNROLL factor=3

							for(int m=0; m<2; m++) {
#pragma HLS UNROLL factor=3

								bool height_bounds_valid_2 = !((H-1+m) <= padding-1 || (H-1+m) >= padding+CONV_IN_HEIGHT);
								bool width_pos1_valid_2 = !((n+W) <= padding-1 || (n+W) >= padding+CONV_IN_WIDTH);
								bool combined_cond_2 = height_bounds_valid_2 && width_pos1_valid_2;

								if(combined_cond_2) {
									float temp_mult2_1 = IFM[ks][H-1+m-padding][n+W-padding] * F[C][ks][m+K-2][n];
									float temp_mult2_2 = IFM[ks][H-1+m-padding][n+W-padding] * F[C][ks][m+K-1][n];



#pragma HLS bind_op variable=temp_mult2_1 op=fmul impl=dsp
#pragma HLS bind_op variable=temp_mult2_2 op=fmul impl=dsp



									local_2_ofm_1 += temp_mult2_1;
									if(m == 0) {
										local_2_ofm_0 += temp_mult2_2;
									}
								}
							}
						}
					} // case-1 end
					else { // case-2 start
						for(int j=0; j<K; j++) {
#pragma HLS UNROLL factor=3

							for(int i=0; i<K; i++) {
#pragma HLS UNROLL factor=3

								bool width_bounds_valid_2 = !((W+j) <= padding-1 || (W+j) >= CONV_IN_WIDTH+padding);
								bool height_pos1_valid_2 = !((i+H-K) <= padding-1 || (i+H-K) >= CONV_IN_HEIGHT+padding);
								bool height_pos2_valid_2 = !((i+H-K+1) <= padding-1 || (i+H-K+1) >= CONV_IN_HEIGHT+padding);
								bool height_pos3_valid_2 = !((i+H-K+2) <= padding-1 || (i+H-K+2) >= CONV_IN_HEIGHT+padding);
								bool height_pos4_valid_2 = !((i+H-K+3) <= padding-1 || (i+H-K+3) >= CONV_IN_HEIGHT+padding);

								bool is_width_boundary_2 = (W == 0);
								bool filter_pos_valid_k2_2 = (i <= K-2);
								bool filter_pos_valid_k3_2 = (i <= K-3);

								if(width_bounds_valid_2) {
									float mult2_result_0 = IFM[ks][i+H-K-padding][W+j-padding] * F[C][ks][i][j];
									float mult2_result_1 = IFM[ks][i+H-K+1-padding][W+j-padding] * F[C][ks][i][j];
									float mult2_result_2 = IFM[ks][i+H-K+2-padding][W+j-padding] * F[C][ks][i][j];
									float mult2_result_3 = IFM[ks][i+H-K+3-padding][W+j-padding] * F[C][ks][i][j];



#pragma HLS bind_op variable=mult2_result_0 op=fmul impl=dsp
#pragma HLS bind_op variable=mult2_result_1 op=fmul impl=dsp
#pragma HLS bind_op variable=mult2_result_2 op=fmul impl=dsp
#pragma HLS bind_op variable=mult2_result_3 op=fmul impl=dsp



									if(height_pos1_valid_2) {
										local_2_ofm_0 += mult2_result_0;
									}

									if(height_pos2_valid_2) {
										local_2_ofm_1 += mult2_result_1;
									}

									if(is_width_boundary_2) {
										if(filter_pos_valid_k2_2) {
											if(height_pos3_valid_2) {
												local_2_ofm_2 += mult2_result_2;
											}

											if(filter_pos_valid_k3_2) {
												if(height_pos4_valid_2) {
													local_2_ofm_3 += mult2_result_3;
												}
											}
										}
									}
								}
							}
						}
					} // case-2 end
				}

				// Final writes to OFM
				float temp2_add_0 = OFM[C][H-K][W] + local_2_ofm_0;
				float temp2_add_1 = OFM[C][H-K+1][W] + local_2_ofm_1;
				float temp2_add_2 = OFM[C][H-K+2][W] + local_2_ofm_2;
				float temp2_add_3 = OFM[C][H-K+3][W] + local_2_ofm_3;



#pragma HLS bind_op variable=temp2_add_0 op=fadd impl=dsp
#pragma HLS bind_op variable=temp2_add_1 op=fadd impl=dsp
#pragma HLS bind_op variable=temp2_add_2 op=fadd impl=dsp
#pragma HLS bind_op variable=temp2_add_3 op=fadd impl=dsp



				OFM[C][H-K][W] = temp2_add_0;
				OFM[C][H-K+1][W] = temp2_add_1;

				if(W == 0) {
					OFM[C][H-K+2][W] = temp2_add_2;
					OFM[C][H-K+3][W] = temp2_add_3;
				}
			} // C loop
		} // W loop
	} // H loop-2 ends
}
