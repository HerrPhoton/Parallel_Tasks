#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ARR_LEN 10000000
#define PI 3.1415926535897932384626433832795
#define TYPE double

int main() {

	TYPE* arr = (TYPE*)malloc(ARR_LEN * sizeof(TYPE));
	TYPE sum = 0;

#pragma acc data create(arr[0:ARR_LEN])
	{
#pragma acc kernels
		for (int i = 0; i < ARR_LEN; i++)
			arr[i] = sin(i * 2 * PI / ARR_LEN);

#pragma acc data copy(sum)
#pragma acc kernels
		for (int i = 0; i < ARR_LEN; i++)
			sum += arr[i];
	}
		
	printf("sum: %0.18lf\n", sum);

	free(arr);
	return 0;
}






