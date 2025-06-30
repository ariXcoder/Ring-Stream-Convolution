
#include <iostream>
#include "hls_stream.h"

#define CONV_IN_HEIGHT    14
#define CONV_IN_WIDTH     14
#define CONV_IN_CHANNELS_2  1 // same as Qn ( for Tn=4 case , for total 6 ifm , 2 ) , for vggnet -	Qn=0
#define CONV_OUT_CHANNELS_COM 1
#define KERNEL_SIZE       5
#define CONV_STRIDE       1
#define KERNEL_CHANNELS_COM   1     // same as the no. of CONV_OUT_CHANNELS_COM
#define padding           0     // user defined, not an input argument

// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)  // 6
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)    // 6



void padded_ring_2_conv2d(
float IFM[CONV_IN_CHANNELS_2][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS_COM][CONV_IN_CHANNELS_2][KERNEL_SIZE][KERNEL_SIZE],
/*float bias[KERNEL_CHANNELS],*/
float OFM[CONV_OUT_CHANNELS_COM][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
){
	/// Array partitioning for input feature maps - complete partition for parallel access




	/*
			#pragma HLS ARRAY_PARTITION variable=IFM complete dim=1
			#pragma HLS ARRAY_PARTITION variable=IFM cyclic factor=2 dim=2
			#pragma HLS ARRAY_PARTITION variable=IFM cyclic factor=2 dim=3

			// Array partitioning for filters - complete partition for kernel dimensions
			#pragma HLS ARRAY_PARTITION variable=F complete dim=1
			#pragma HLS ARRAY_PARTITION variable=F complete dim=2
			#pragma HLS ARRAY_PARTITION variable=F complete dim=3
			#pragma HLS ARRAY_PARTITION variable=F complete dim=4

			// Array partitioning for output feature maps
			#pragma HLS ARRAY_PARTITION variable=OFM complete dim=1
			#pragma HLS ARRAY_PARTITION variable=OFM cyclic factor=2 dim=2
			#pragma HLS ARRAY_PARTITION variable=OFM cyclic factor=2 dim=3
	*/





		  // Initialize output feature map to zero
			init_output:
			for (int h = 0; h < CONV_OUT_HEIGHT; h++) {
				#pragma HLS LOOP_TRIPCOUNT min=6 max=6
				for (int w = 0; w < CONV_OUT_WIDTH; w++) {
					//#pragma HLS LOOP_TRIPCOUNT min=6 max=6
					#pragma HLS PIPELINE II=1

					for (int oc = 0; oc < CONV_OUT_CHANNELS_COM; oc++) {
						#pragma HLS LOOP_TRIPCOUNT min=1 max=1
						OFM[oc][h][w] = 0;
					}
				}
			}


			ring_conv:

				for(int H=KERNEL_SIZE-1; H<CONV_IN_HEIGHT+2*padding; H=H+CONV_STRIDE){
	//#pragma HLS UNROLL factor=28
	//#pragma HLS LOOP_TRIPCOUNT min=8 max=8
	//#pragma HLS pipeline II=1

					if(H%2==0){
						for(int W=KERNEL_SIZE-1; W<CONV_IN_WIDTH+2*padding; W=W+CONV_STRIDE){
	//#pragma HLS LOOP_TRIPCOUNT min=8 max=8
	//#pragma HLS UNROLL factor=7
							for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1){
	//#pragma HLS LOOP_TRIPCOUNT min=1 max=1

								// this loop is for multiple channels of IFM and single 3D kernel
								for(int ks=0;ks<CONV_IN_CHANNELS_2;ks++){
	//#pragma HLS LOOP_TRIPCOUNT min=4 max=4
	//#pragma HLS PIPELINE II=1
	//#pragma HLS UNROLL factor=1
									float local=0.0f;

									// Partially unroll inner loops to balance performance and resources
									for(int i=0; i<KERNEL_SIZE;i=i+1){
	//#pragma HLS LOOP_TRIPCOUNT min=3 max=3
	//#pragma HLS UNROLL factor=5
										for(int j=0; j<KERNEL_SIZE; j=j+1){
	//#pragma HLS LOOP_TRIPCOUNT min=3 max=3
	//#pragma HLS UNROLL factor=5

											int wtemp=W+j-KERNEL_SIZE+1; // index for width of ifm
											int htemp=H+1+i-KERNEL_SIZE; // index for hieght of ifm

											if(wtemp<=padding-1 || wtemp >=CONV_IN_WIDTH+padding){
												local+=0.0f;
											}
											else if(htemp<=padding-1 || htemp >=CONV_IN_HEIGHT+padding){
												local+=0.0f;
											}
											else{
												local+=IFM[ks][H+1+i-KERNEL_SIZE-padding][W+j-KERNEL_SIZE+1-padding]*F[C][ks][i][j];
											}
										}

									}
									// add all the single-channel values to main OFM
									OFM[C][H+1-KERNEL_SIZE][W+1-KERNEL_SIZE]+=local;
							   }
							}
						}
					}

					else {

						for(int W=CONV_IN_WIDTH-KERNEL_SIZE+padding+1; W>=0; W=W-CONV_STRIDE){
							//#pragma HLS LOOP_TRIPCOUNT min=8 max=8

							for(int C=0; C<CONV_OUT_CHANNELS_COM; C=C+1){
								//#pragma HLS LOOP_TRIPCOUNT min=1 max=1

								// this loop is for multiple channels of IFM and single 3D kernel
								for(int ks=0;ks<CONV_IN_CHANNELS_2;ks++){
									//#pragma HLS LOOP_TRIPCOUNT min=4 max=4
									//#pragma HLS PIPELINE II=1
	//#pragma HLS UNROLL factor=1
									float local2=0.0f;

									// Partially unroll inner loops to balance performance and resources
									for(int i=0; i<KERNEL_SIZE;i=i+1){
	//#pragma HLS LOOP_TRIPCOUNT min=3 max=3
	//#pragma HLS UNROLL factor=5
										for(int j=0; j<KERNEL_SIZE; j=j+1){
	//#pragma HLS LOOP_TRIPCOUNT min=3 max=3
	//#pragma HLS UNROLL factor=5

											int wtemp1=W+j; // index for width of ifm
											int htemp1=H+1+i-KERNEL_SIZE; // index for hieght of ifm

											if(wtemp1<=padding-1 || wtemp1 >=CONV_IN_WIDTH+padding){
												local2+=0.0f;
											}
											else if(htemp1<=padding-1 || htemp1 >=CONV_IN_HEIGHT+padding){
												local2+=0.0f;
											}
											else{
												local2+=IFM[ks][H+1+i-KERNEL_SIZE-padding][W+j-padding]*F[C][ks][i][j];

											}
										}

									}
									// add all the single-channel values to main OFM
									OFM[C][H+1-KERNEL_SIZE][W]+=local2;
								}
							}
						}
					}
				}

			};


