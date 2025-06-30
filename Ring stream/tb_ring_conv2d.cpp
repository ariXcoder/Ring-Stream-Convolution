#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

#define CONV_IN_HEIGHT    8
#define CONV_IN_WIDTH     8
#define CONV_IN_CHANNELS  6
#define CONV_OUT_CHANNELS 6
#define KERNEL_SIZE       3
#define CONV_STRIDE       1
#define KERNEL_CHANNELS   6     // same as the no. of CONV_OUT_CHANNELS

#define padding           0     // user defined, not an input argument

// Calculate output dimensions
#define CONV_OUT_HEIGHT ((CONV_IN_HEIGHT - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)  // 6
#define CONV_OUT_WIDTH ((CONV_IN_WIDTH - KERNEL_SIZE+2*padding) / CONV_STRIDE + 1)    // 6


#define KERNEL_CHANNELS_COM   1
#define CONV_OUT_CHANNELS_COM 1


#define CONV_IN_CHANNELS_1 3


/*
void ring_1_conv2d(
float IFM[4][8][8],
float F[1][4][3][3],       //3x3x2x4 - 4 2 3 3
float OFM[1][6][6]
);


#define CONV_IN_CHANNELS_2 2
void ring_2_conv2d(
		float IFM[2][8][8],
		float F[1][2][3][3],       //3x3x2x4 - 4 2 3 3
		float OFM[1][6][6]
		);
*/

/*

void padded_ring_1_conv2d(
float IFM[4][8][8],
float F[1][4][3][3],       //3x3x2x4 - 4 2 3 3
float OFM[1][8][8]
);


#define CONV_IN_CHANNELS_2 0
void padded_ring_2_conv2d(
		float IFM[2][8][8],
		float F[1][2][3][3],       //3x3x2x4 - 4 2 3 3
		float OFM[1][8][8]
		);
*/



/*
void tile_ring_conv2d(
float IFM[CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS][CONV_IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],       //3x3x2x4 - 4 2 3 3
float OFM[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
);
*/


void padded_ring_conv2d(
float IFM[CONV_IN_CHANNELS][CONV_IN_HEIGHT][CONV_IN_WIDTH],
float F[KERNEL_CHANNELS][CONV_IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],

float OFM[CONV_OUT_CHANNELS][CONV_OUT_HEIGHT][CONV_OUT_WIDTH]
);


int main(){


	/*float IFM[1][8][8] = {
		{
			{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
			{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
			{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
			{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
			{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
			{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
			{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
			{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
			//W=1,W=2,W=3 ..................W=7, W=8
		}
	};*/

	/*float F[1][1][3][3]={
		{
			{
				{ 1.0f, 0.0f, -1.0f },
				{ 2.0f, 0.0f, -2.0f },
				{ 1.0f, 0.0f, -1.0f }
			}

		 }
	};*/
			//C  H  W
	float IFM[6][8][8]={
			{ // C=1
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
				{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
				{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
				{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
				{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
				{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
				//W=1,W=2,W=3 ..................W=7, W=8                        // 2 -channel IFM case
			}
			,{ // C=2
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
				{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
				{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
				{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
				{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
				{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
				{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
				//W=1,W=2,W=3 ..................W=7, W=8
			}
			,{ // C=3
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
							{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
							{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
							{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
							{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
							{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
							//W=1,W=2,W=3 ..................W=7, W=8
						}
			,{ // C=4
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
							{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
							{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
							{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
							{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
							{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
							//W=1,W=2,W=3 ..................W=7, W=8
						}
			,{ // C=5
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
							{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
							{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
							{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
							{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
							{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
							//W=1,W=2,W=3 ..................W=7, W=8
						}
			,{ // C=6
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-1
							{4.0f,5.0f,6.0f,4.0f,5.0f,6.0f,3.0f,4.0f},  // H-2
							{7.0f,8.0f,9.0f,7.0f,8.0f,9.0f,5.0f,6.0f},  // H-3
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,1.0f,2.0f},  // H-4
							{1.0f,3.0f,4.0f,2.0f,3.0f,4.0f,3.0f,4.0f},  // H-5
							{2.0f,4.0f,5.0f,3.0f,4.0f,5.0f,5.0f,6.0f},  // H-6
							{1.0f,2.0f,3.0f,1.0f,2.0f,3.0f,3.0f,4.0f},  // H-7
							{4.0f,5.0f,6.0f,2.0f,3.0f,4.0f,4.0f,5.0f}   // H-8
							//W=1,W=2,W=3 ..................W=7, W=8
						}


	};
	//      C  N  H  W   C- total no. of 3d kernels
	float F[6][6][3][3] = {
	  {//C-1

		 {//N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 }

		 ,{//N-2
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }
                                                       // 2 no. of 2-channeled kernel
		  }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }

	  }

	  ,{//C-2

		 { //N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 },
		 {// N-2
				 { 1.0f, 0.0f, -1.0f },
				 			{ 2.0f, 0.0f, -2.0f },
				 			{ 1.0f, 0.0f, -1.0f }

				 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
	  }
	  ,{//C-3

		 { //N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 },
		 {// N-2
				 { 1.0f, 0.0f, -1.0f },
				 			{ 2.0f, 0.0f, -2.0f },
				 			{ 1.0f, 0.0f, -1.0f }

				 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
	  	  }
	  ,{//C-4

		 { //N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 },
		 {// N-2
				 { 1.0f, 0.0f, -1.0f },
				 			{ 2.0f, 0.0f, -2.0f },
				 			{ 1.0f, 0.0f, -1.0f }

				 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
	  	  }
	  ,{//C-5

		 { //N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 },
		 {// N-2
				 { 1.0f, 0.0f, -1.0f },
				 			{ 2.0f, 0.0f, -2.0f },
				 			{ 1.0f, 0.0f, -1.0f }

				 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
	   }

	  ,{//C-6

		 { //N-1
			{ 1.0f, 0.0f, -1.0f },
			{ 2.0f, 0.0f, -2.0f },
			{ 1.0f, 0.0f, -1.0f }

		 },
		 {// N-2
				 { 1.0f, 0.0f, -1.0f },
				{ 2.0f, 0.0f, -2.0f },
				{ 1.0f, 0.0f, -1.0f }

				 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
		 ,{//N-2
					{ 1.0f, 0.0f, -1.0f },
					{ 2.0f, 0.0f, -2.0f },
					{ 1.0f, 0.0f, -1.0f }
		                                                       // 2 no. of 2-channeled kernel
				 		 }
	   }

	};

	//float bias[1]={0.0f};

	float OFM[6][6][6];

	padded_ring_conv2d(IFM,F,OFM);  // test case IFM : 8x8x6 , F: 3x3x6x6 , OFM : 6X6X6  Tn=4,Tm=2

	cout<<"OUTPUT VALUE :"<<endl;
	for(int c=0;c<CONV_OUT_CHANNELS;c=c+1){
		cout<<"CHANNEL-"<<c+1<<endl;
		for(int i=0;i<CONV_OUT_HEIGHT;i=i+1){

			for(int j=0;j<CONV_OUT_WIDTH;j=j+1){

					cout<<"  "<<OFM[c][i][j]<<"  ";
			}
			cout<<endl;

		}
	}

	return 0;
}
