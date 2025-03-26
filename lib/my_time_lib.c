#include "../include/my_time_lib.h"
#include <math.h>

// Put here the implementation of mu_fn and sigma_fn
double mu_fn(double *v, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum / n;
}

double sigma_fn(double *v, double mu, int len) {
    double sigma = 0.0;
    for (int i=0; i<len; i++) {
        sigma += ((double)v[i] - mu)*((double)v[i] - mu);
    }
    sigma /= (double)len;

    return(sigma);
}

double gm_fn(double *v, int n) {
    double product = 1.0;
    for (int i = 0; i < n; i++) {
        product *= v[i] > 0 ? v[i] : 1.0;  // Use absolute values
    }
    double res = pow(product, 1.0 / n);
    return res;
}
// -------------------------------------------------
