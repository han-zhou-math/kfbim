/*=============================================================================
*   
*   Filename : CauchyProblemSolvers.h
*   Creator : Han Zhou
*   Date : 02/10/22
*   Description : 
*
=============================================================================*/

#ifndef _CAUCHYPROBLEMSOLVERS_H
#define _CAUCHYPROBLEMSOLVERS_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

#include <eigen3/Eigen/Dense>
 
#include "MathTools.h"
#include "Variables.h"
#include "QR.h"

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// collocation with quadratic polynomial (3-th order)
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void solve2DCauchyProblem2(const double bdry1_crd[3][2], 
															 	  const double bdry2_crd[2][2], 
															 	  const double bdry2_nml[2][2], 
															 	  const double bulk_crd[2],  
															 	  const double u[3],	
																	const double un[2],
															 	  double Lu, const double center[2], 
															 	  double kappa, double h, 
																	double c[6])
{
	double h2 = h * h;

	double kap_h2 = kappa * h2;

	double mat[6][6], b[6];

	int m = 0;

	// [u] = phi

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][0] = 1.0;

		mat[m][1] = dx;
		mat[m][2] = dy;

		mat[m][3] = 0.5*dx*dx;
		mat[m][4] = 0.5*dy*dy;
		mat[m][5] = dx*dy;

		b[m] = u[l];

		m++;
	}

	// [u_n] = psi

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][0] = 0.0;

		mat[m][1] = nx;
		mat[m][2] = ny;

		mat[m][3] = dx*nx;
		mat[m][4] = dy*ny;
		mat[m][5] = nx*dy+dx*ny;

		b[m] = un[l] * h;

		m++;
	}


	// [- (u_xx+u_yy) + kappa * u] = [f]

	{

		double x0 = bulk_crd[0];
		double y0 = bulk_crd[1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][0] = kap_h2*1.0;

		mat[m][1] = kap_h2*dx;
		mat[m][2] = kap_h2*dy;

		mat[m][3] = kap_h2*0.5*dx*dx-1.0;
		mat[m][4] = kap_h2*0.5*dy*dy-1.0;
		mat[m][5] = kap_h2*dx*dy;

		b[m] = Lu * h2;

		m++;
	}

	preConditionSystem<6>(mat, b);
	bool status = solveByQRdecomposition<6>(mat, b, c, 6);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	for(int i = 1; i < 3; i++){
		c[i] /= h;
	}
	for(int i = 3; i < 6; i++){
		c[i] /= h2;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// collocation with quartic polynomial (5-th order)
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void solve2DCauchyProblem4(const double bdry1_crd[5][2], 
															 		const double bdry2_crd[4][2], 
															 		const double bdry2_nml[4][2], 
															 		const double bulk_crd[6][2],  
															 		const double u[5],	const double un[4],
															 		const double Lu[6], const double center[2], 
															 		double kappa, double h, double c[15])
{

	double h2 = h * h;
	double h3 = h2 * h;
	double h4 = h2 * h2;

	double kap_h2 = kappa * h2;

	double mat[15][15], b[15];

	int m = 0;

	// [u] = phi

	for(int l = 0; l < 5; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][0] = 1.0;

		mat[m][1] = dx;
		mat[m][2] = dy;

		mat[m][3] = 0.5*dx*dx;
		mat[m][4] = 0.5*dy*dy;
		mat[m][5] = dx*dy;

		mat[m][6] = dx*dx*dx/6.0;
		mat[m][7] = dy*dy*dy/6.0;
		mat[m][8] = 0.5*dx*dx*dy;
		mat[m][9] = 0.5*dy*dy*dx;

		mat[m][10] = dx*dx*dx*dx/24.0;
		mat[m][11] = dy*dy*dy*dy/24.0;
		mat[m][12] = dx*dx*dx*dy/6.0;
		mat[m][13] = dy*dy*dy*dx/6.0;
		mat[m][14] = 0.25*dx*dx*dy*dy;

		b[m] = u[l];

		m++;
	}

	// [u_n] = psi

	for(int l = 0; l < 4; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][0] = 0.0;

		mat[m][1] = nx;
		mat[m][2] = ny;

		mat[m][3] = dx*nx;
		mat[m][4] = dy*ny;
		mat[m][5] = nx*dy+dx*ny;

		mat[m][6] = 0.5*nx*dx*dx;
		mat[m][7] = 0.5*ny*dy*dy;
		mat[m][8] = nx*dx*dy+0.5*ny*dx*dx;
		mat[m][9] = ny*dy*dx+0.5*nx*dy*dy;

		mat[m][10] = nx*dx*dx*dx/6.0;
		mat[m][11] = ny*dy*dy*dy/6.0;
		mat[m][12] = dx*dx*dx*ny/6.0+nx*dx*dx*dy/2.0;
		mat[m][13] = dy*dy*dy*nx/6.0+ny*dy*dy*dx/2.0;
		mat[m][14] = 0.5*(nx*dx*dy*dy+dx*dx*ny*dy);

		b[m] = un[l] * h;

		m++;
	}


	// [- (u_xx+u_yy+u_zz) + kappa * u] = [f]

	for(int l = 0; l < 6; l++){

		double x0 = bulk_crd[l][0];
		double y0 = bulk_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][0] = kap_h2*1.0;

		mat[m][1] = kap_h2*dx;
		mat[m][2] = kap_h2*dy;

		mat[m][3] = kap_h2*0.5*dx*dx-1.0;
		mat[m][4] = kap_h2*0.5*dy*dy-1.0;
		mat[m][5] = kap_h2*dx*dy;

		mat[m][6] = kap_h2*dx*dx*dx/6.0-dx;
		mat[m][7] = kap_h2*dy*dy*dy/6.0-dy;
		mat[m][8] = kap_h2*0.5*dx*dx*dy-dy;
		mat[m][9] = kap_h2*0.5*dy*dy*dx-dx;

		mat[m][10] = kap_h2*dx*dx*dx*dx/24.0-0.5*dx*dx;
		mat[m][11] = kap_h2*dy*dy*dy*dy/24.0-0.5*dy*dy;
		mat[m][12] = kap_h2*dx*dx*dx*dy/6.0-dx*dy;
		mat[m][13] = kap_h2*dy*dy*dy*dx/6.0-dy*dx;
		mat[m][14] = kap_h2*0.25*dx*dx*dy*dy-0.5*(dy*dy+dx*dx);

		b[m] = Lu[l] * h2;

		m++;
	}

	preConditionSystem<15>(mat, b);
	bool status = solveByQRdecomposition<15>(mat, b, c, 15);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	for(int i = 1; i < 3; i++){
		c[i] /= h;
	}
	for(int i = 3; i < 6; i++){
		c[i] /= h2;
	}
	for(int i = 6; i < 10; i++){
		c[i] /= h3;
	}
	for(int i = 10; i < 15; i++){
		c[i] /= h4;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// collocation with quadratic polynomial (3-th order)
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void solve3DCauchyProblem2(const double bdry1_crd[6][3], 
															 	  const double bdry2_crd[3][3], 
															 	  const double bdry2_nml[3][3], 
															 	  const double bulk_crd[3],  
															 	  const double u[6],	const double un[3],
															 	  double Lu, const double center[3], 
															 	  double kappa, double h, 
																	double c[10])
{
	double h2 = h * h;

	double kap_h2 = kappa * h2;

	double mat[10][10], b[10];

	int m = 0;

	// [u] = phi

	for(int l = 0; l < 6; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];
		double z0 = bdry1_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		mat[m][0] = 1.0;

		mat[m][1] = dx;
		mat[m][2] = dy;
		mat[m][3] = dz;

		mat[m][4] = 0.5*dx*dx;
		mat[m][5] = 0.5*dy*dy;
		mat[m][6] = 0.5*dz*dz;
		mat[m][7] = dy*dz;
		mat[m][8] = dz*dx;
		mat[m][9] = dx*dy;

		b[m] = u[l];

		m++;
	}

	// [u_n] = psi

	for(int l = 0; l < 3; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];
		double z0 = bdry2_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];
		double nz = bdry2_nml[l][2];

		mat[m][0] = 0.0;

		mat[m][1] = nx;
		mat[m][2] = ny;
		mat[m][3] = nz;

		mat[m][4] = dx*nx;
		mat[m][5] = dy*ny;
		mat[m][6] = dz*nz;
		mat[m][7] = ny*dz+dy*nz;
		mat[m][8] = nz*dx+dz*nx;
		mat[m][9] = nx*dy+dx*ny;

		b[m] = un[l] * h;

		m++;
	}


	// [- (u_xx+u_yy+u_zz) + kappa * u] = [f]

	{

		double x0 = bulk_crd[0];
		double y0 = bulk_crd[1];
		double z0 = bulk_crd[2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		mat[m][0] = kap_h2*1.0;

		mat[m][1] = kap_h2*dx;
		mat[m][2] = kap_h2*dy;
		mat[m][3] = kap_h2*dz;

		mat[m][4] = kap_h2*0.5*dx*dx-1.0;
		mat[m][5] = kap_h2*0.5*dy*dy-1.0;
		mat[m][6] = kap_h2*0.5*dz*dz-1.0;
		mat[m][7] = kap_h2*dy*dz;
		mat[m][8] = kap_h2*dz*dx;
		mat[m][9] = kap_h2*dx*dy;

		b[m] = Lu * h2;

		m++;
	}

	preConditionSystem<10>(mat, b);
	bool status = solveByQRdecomposition<10>(mat, b, c, 10);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	for(int i = 1; i < 4; i++){
		c[i] /= h;
	}
	for(int i = 4; i < 10; i++){
		c[i] /= h2;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// collocation with quartic polynomial (5-th order)
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void solve3DCauchyProblem4(const double bdry1_crd[15][3], 
															 		const double bdry2_crd[10][3], 
															 		const double bdry2_nml[10][3], 
															 		const double bulk_crd[10][3],  
															 		const double u[15],	const double un[10],
															 		const double Lu[10], const double center[3], 
															 		double kappa, double h, 
																	double c[35])
{
	double h2 = h * h;
	double h3 = h2 * h;
	double h4 = h2 * h2;

	double kap_h2 = kappa * h2;

	double mat[35][35], b[35];

	int m = 0;

	// [u] = phi

	for(int l = 0; l < 15; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];
		double z0 = bdry1_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		mat[m][0] = 1.0;

		mat[m][1] = dx;
		mat[m][2] = dy;
		mat[m][3] = dz;

		mat[m][4] = 0.5*dx*dx;
		mat[m][5] = 0.5*dy*dy;
		mat[m][6] = 0.5*dz*dz;
		mat[m][7] = dy*dz;
		mat[m][8] = dz*dx;
		mat[m][9] = dx*dy;

		mat[m][10] = dx*dx*dx/6.0;
		mat[m][11] = dy*dy*dy/6.0;
		mat[m][12] = dz*dz*dz/6.0;
		mat[m][13] = 0.5*dx*dx*dy;
		mat[m][14] = 0.5*dx*dx*dz;
		mat[m][15] = 0.5*dy*dy*dx;
		mat[m][16] = 0.5*dy*dy*dz;
		mat[m][17] = 0.5*dz*dz*dx;
		mat[m][18] = 0.5*dz*dz*dy;
		mat[m][19] = dx*dy*dz;

		mat[m][20] = dx*dx*dx*dx/24.0;
		mat[m][21] = dy*dy*dy*dy/24.0;
		mat[m][22] = dz*dz*dz*dz/24.0;
		mat[m][23] = dx*dx*dx*dy/6.0;
		mat[m][24] = dx*dx*dx*dz/6.0;
		mat[m][25] = dy*dy*dy*dx/6.0;
		mat[m][26] = dy*dy*dy*dz/6.0;
		mat[m][27] = dz*dz*dz*dx/6.0;
		mat[m][28] = dz*dz*dz*dy/6.0;
		mat[m][29] = 0.25*dx*dx*dy*dy;
		mat[m][30] = 0.25*dy*dy*dz*dz;
		mat[m][31] = 0.25*dz*dz*dx*dx;
		mat[m][32] = 0.5*dx*dx*dy*dz;
		mat[m][33] = 0.5*dy*dy*dz*dx;
		mat[m][34] = 0.5*dz*dz*dx*dy;

		b[m] = u[l];

		m++;
	}

	// [u_n] = psi

	for(int l = 0; l < 10; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];
		double z0 = bdry2_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];
		double nz = bdry2_nml[l][2];

		mat[m][0] = 0.0;

		mat[m][1] = nx;
		mat[m][2] = ny;
		mat[m][3] = nz;

		mat[m][4] = dx*nx;
		mat[m][5] = dy*ny;
		mat[m][6] = dz*nz;
		mat[m][7] = ny*dz+dy*nz;
		mat[m][8] = nz*dx+dz*nx;
		mat[m][9] = nx*dy+dx*ny;

		mat[m][10] = 0.5*nx*dx*dx;
		mat[m][11] = 0.5*ny*dy*dy;
		mat[m][12] = 0.5*nz*dz*dz;
		mat[m][13] = nx*dx*dy+0.5*ny*dx*dx;
		mat[m][14] = nx*dx*dz+0.5*nz*dx*dx;
		mat[m][15] = ny*dy*dx+0.5*nx*dy*dy;
		mat[m][16] = ny*dy*dz+0.5*nz*dy*dy;
		mat[m][17] = nz*dz*dx+0.5*nx*dz*dz;
		mat[m][18] = nz*dz*dy+0.5*ny*dz*dz;
		mat[m][19] = nx*dy*dz+dx*ny*dz+dx*dy*nz;

		mat[m][20] = nx*dx*dx*dx/6.0;
		mat[m][21] = ny*dy*dy*dy/6.0;
		mat[m][22] = nz*dz*dz*dz/6.0;
		mat[m][23] = dx*dx*dx*ny/6.0+nx*dx*dx*dy/2.0;
		mat[m][24] = dx*dx*dx*nz/6.0+nx*dx*dx*dz/2.0;
		mat[m][25] = dy*dy*dy*nx/6.0+ny*dy*dy*dx/2.0;
		mat[m][26] = dy*dy*dy*nz/6.0+ny*dy*dy*dz/2.0;
		mat[m][27] = dz*dz*dz*nx/6.0+nz*dz*dz*dx/2.0;
		mat[m][28] = dz*dz*dz*ny/6.0+nz*dz*dz*dy/2.0;
		mat[m][29] = 0.5*(nx*dx*dy*dy+dx*dx*ny*dy);
		mat[m][30] = 0.5*(ny*dy*dz*dz+dy*dy*nz*dz);
		mat[m][31] = 0.5*(nz*dz*dx*dx+dz*dz*nx*dx);
		mat[m][32] = nx*dx*dy*dz+0.5*(dx*dx*ny*dz+dx*dx*dy*nz);
		mat[m][33] = ny*dy*dz*dx+0.5*(dy*dy*nz*dx+dy*dy*dz*nx);
		mat[m][34] = nz*dz*dx*dy+0.5*(dz*dz*nx*dy+dz*dz*dx*ny);

		b[m] = un[l] * h;

		m++;
	}


	// [- (u_xx+u_yy+u_zz) + kappa * u] = [f]

	for(int l = 0; l < 10; l++){

		double x0 = bulk_crd[l][0];
		double y0 = bulk_crd[l][1];
		double z0 = bulk_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		mat[m][0] = kap_h2*1.0;

		mat[m][1] = kap_h2*dx;
		mat[m][2] = kap_h2*dy;
		mat[m][3] = kap_h2*dz;

		mat[m][4] = kap_h2*0.5*dx*dx-1.0;
		mat[m][5] = kap_h2*0.5*dy*dy-1.0;
		mat[m][6] = kap_h2*0.5*dz*dz-1.0;
		mat[m][7] = kap_h2*dy*dz;
		mat[m][8] = kap_h2*dz*dx;
		mat[m][9] = kap_h2*dx*dy;

		mat[m][10] = kap_h2*dx*dx*dx/6.0-dx;
		mat[m][11] = kap_h2*dy*dy*dy/6.0-dy;
		mat[m][12] = kap_h2*dz*dz*dz/6.0-dz;
		mat[m][13] = kap_h2*0.5*dx*dx*dy-dy;
		mat[m][14] = kap_h2*0.5*dx*dx*dz-dz;
		mat[m][15] = kap_h2*0.5*dy*dy*dx-dx;
		mat[m][16] = kap_h2*0.5*dy*dy*dz-dz;
		mat[m][17] = kap_h2*0.5*dz*dz*dx-dx;
		mat[m][18] = kap_h2*0.5*dz*dz*dy-dy;
		mat[m][19] = kap_h2*dx*dy*dz;

		mat[m][20] = kap_h2*dx*dx*dx*dx/24.0-0.5*dx*dx;
		mat[m][21] = kap_h2*dy*dy*dy*dy/24.0-0.5*dy*dy;
		mat[m][22] = kap_h2*dz*dz*dz*dz/24.0-0.5*dz*dz;
		mat[m][23] = kap_h2*dx*dx*dx*dy/6.0-dx*dy;
		mat[m][24] = kap_h2*dx*dx*dx*dz/6.0-dx*dz;
		mat[m][25] = kap_h2*dy*dy*dy*dx/6.0-dy*dx;
		mat[m][26] = kap_h2*dy*dy*dy*dz/6.0-dy*dz;
		mat[m][27] = kap_h2*dz*dz*dz*dx/6.0-dz*dx;
		mat[m][28] = kap_h2*dz*dz*dz*dy/6.0-dz*dy;
		mat[m][29] = kap_h2*0.25*dx*dx*dy*dy-0.5*(dy*dy+dx*dx);
		mat[m][30] = kap_h2*0.25*dy*dy*dz*dz-0.5*(dz*dz+dy*dy);
		mat[m][31] = kap_h2*0.25*dz*dz*dx*dx-0.5*(dx*dx+dz*dz);
		mat[m][32] = kap_h2*0.5*dx*dx*dy*dz-dy*dz;
		mat[m][33] = kap_h2*0.5*dy*dy*dz*dx-dz*dx;
		mat[m][34] = kap_h2*0.5*dz*dz*dx*dy-dx*dy;

		b[m] = Lu[l] * h2;

		m++;
	}

	preConditionSystem<35>(mat, b);
	bool status = solveByQRdecomposition<35>(mat, b, c, 35);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	for(int i = 1; i < 4; i++){
		c[i] /= h;
	}
	for(int i = 4; i < 10; i++){
		c[i] /= h2;
	}
	for(int i = 10; i < 20; i++){
		c[i] /= h3;
	}
	for(int i = 20; i < 35; i++){
		c[i] /= h4;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// Stokes Cauchy problem sovler
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*inline void solve2DStokesCauchyProblem2(const double bdry1_crd[3][2], 
															 	  			const double bdry2_crd[2][2], 
															 	  			const double bdry2_nml[2][2], 
															 	  			const double bulk_crd[2],  
															 	  			const double u1[3],	
															 	  			const double u2[3],	
																				const double Sn1[2],
																				const double Sn2[2],
															 	  			double f1, 
															 	  			double f2, 
																				const double center[2], 
															 	  			double h, 
																				double u1j[6],
																				double u2j[6],
																				double pj[3])
{
	double h2 = h * h;

	double mat[15][15], b[15], c[15];

	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = mat[m1][6]  = 1.0;
                
		mat[m][1] = mat[m1][7]  = dx;
		mat[m][2] = mat[m1][8]  = dy;
                
		mat[m][3] = mat[m1][9]  = 0.5*dx*dx;
		mat[m][4] = mat[m1][10] = 0.5*dy*dy;
		mat[m][5] = mat[m1][11] = dx*dy;

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][1] = 2.0*nx;
		mat[m][2] = ny;
		mat[m][3] = 2.0*nx*dx;
		mat[m][4] = ny*dy;
		mat[m][5] = 2.0*nx*dy + ny*dx;

		mat[m][7] = ny;
		mat[m][9] = ny*dx;
		mat[m][11] = ny*dy;

		mat[m][12] = -nx;
		mat[m][13] = -nx*dx;
		mat[m][14] = -nx*dy;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][2] = nx;
		mat[m][4] = nx*dy;
		mat[m][5] = nx*dx;

		mat[m][7] = nx;
		mat[m][8] = 2.0*ny;
		mat[m][9] = nx*dx;
		mat[m][10] = 2.0*ny*dy;
		mat[m][11] = nx*dy+2.0*ny*dx;

		mat[m][12] = -ny;
		mat[m][13] = -ny*dx;
		mat[m][14] = -ny*dy;

		b[m++] = Sn2[l] * h;
	}


	{
		double x0 = bulk_crd[0];
		double y0 = bulk_crd[1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		// div u = 0

		mat[m][8] = mat[m][1] = 1.0;
		mat[m][11] = mat[m][3] = dx;
		mat[m][10] = mat[m][5] = dy;

		b[m++] = 0.0;

		mat[m][3] = mat[m][11] = 1.0;
		b[m++] = 0.0;

		mat[m][5] = mat[m][10] = 1.0;
		b[m++] = 0.0;

		// - \Delta u + \nabla p = f

		mat[m][3] = mat[m][4] = -1.0;
		mat[m][13] = 1.0;
		b[m++] = f1*h2;

		mat[m][9] = mat[m][10] = -1.0;
		mat[m][14] = 1.0;
		b[m++] = f2*h2;
	}

	bool status = solveByQRdecomposition<15>(mat, b, c, 15);
	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(15, 15);
	//for(int i = 0; i < 15; i++){
	//	for(int j = 0; j < 15; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//}
	//std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;

	u1j[0] = c[0];
	u1j[1] = c[1]/h;
	u1j[2] = c[2]/h;
	u1j[3] = c[3]/h2;
	u1j[4] = c[4]/h2;
	u1j[5] = c[5]/h2;

	u2j[0] = c[6];
	u2j[1] = c[7]/h;
	u2j[2] = c[8]/h;
	u2j[3] = c[9]/h2;
	u2j[4] = c[10]/h2;
	u2j[5] = c[11]/h2;

	pj[0] = c[12]/h;
	pj[1] = c[13]/h2;
	pj[2] = c[14]/h2;

}*/

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void 
solve2DStokesCauchyProblem2(const double bdry1_crd[3][2],  // Dirichlet BC nodes
														const double bdry2_crd[2][2], // Neumann BC nodes
														const double bdry2_nml[2][2], // Neumann BC nodes
														const double bulk1_crd[2],  	// PDE1 nodes
														const double bulk2_crd[3][2], // div free nodes
														const double u1[3],						// Dirichlet1 BC val
														const double u2[3],						// Dirichlet2 BC val
														const double Sn1[2],					// Neumann1 BC val
														const double Sn2[2],					// Neumann2 BC val
														double f1, 										// RHS1 val
														double f2, 										// RHS2 val
														const double center[2], 			// patch center
														double mu, 										// viscosity
														double h, 										// grid size
														double u1j[6],								// u1 sol
														double u2j[6],								// u1 sol
														double pj[3])									// p sol
{
	double h2 = h * h;

	double mat[15][15], b[15], c[15];

	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = mat[m1][6]  = 1.0;
                
		mat[m][1] = mat[m1][7]  = dx;
		mat[m][2] = mat[m1][8]  = dy;
                
		mat[m][3] = mat[m1][9]  = 0.5*dx*dx;
		mat[m][4] = mat[m1][10] = 0.5*dy*dy;
		mat[m][5] = mat[m1][11] = dx*dy;

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][1] = mu*2.0*nx;
		mat[m][2] = mu*ny;
		mat[m][3] = mu*2.0*nx*dx;
		mat[m][4] = mu*ny*dy;
		mat[m][5] = mu*(2.0*nx*dy + ny*dx);

		mat[m][7] = mu*ny;
		mat[m][9] = mu*ny*dx;
		mat[m][11] = mu*ny*dy;

		mat[m][12] = -nx;
		mat[m][13] = -nx*dx;
		mat[m][14] = -nx*dy;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][2] = mu*nx;
		mat[m][4] = mu*nx*dy;
		mat[m][5] = mu*nx*dx;

		mat[m][7] = mu*nx;
		mat[m][8] = mu*2.0*ny;
		mat[m][9] = mu*nx*dx;
		mat[m][10] = mu*2.0*ny*dy;
		mat[m][11] = mu*(nx*dy+2.0*ny*dx);

		mat[m][12] = -ny;
		mat[m][13] = -ny*dx;
		mat[m][14] = -ny*dy;

		b[m++] = Sn2[l] * h;

	}

	// - mu \Delta u + \nabla p = f

	{
		double x0 = bulk1_crd[0];
		double y0 = bulk1_crd[1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][3] = mat[m][4] = -1.0*mu;
		mat[m][13] = 1.0;

		b[m++] = f1*h2;

		mat[m][9] = mat[m][10] = -1.0*mu;
		mat[m][14] = 1.0;

		b[m++] = f2*h2;
	}

	// div u = 0

	for(int l = 0; l < 3; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][8] = mat[m][1] = 1.0;
		mat[m][11] = mat[m][3] = dx;
		mat[m][10] = mat[m][5] = dy;

		b[m++] = 0.0;
	}

	assert(m == 15);

	preConditionSystem<15>(mat, b);
	bool status = solveByQRdecomposition<15>(mat, b, c, 15);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(15, 15);
	//for(int i = 0; i < 15; i++){
	//	for(int j = 0; j < 15; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//}
	//std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;

	u1j[0] = c[0];
	u1j[1] = c[1]/h;
	u1j[2] = c[2]/h;
	u1j[3] = c[3]/h2;
	u1j[4] = c[4]/h2;
	u1j[5] = c[5]/h2;

	u2j[0] = c[6];
	u2j[1] = c[7]/h;
	u2j[2] = c[8]/h;
	u2j[3] = c[9]/h2;
	u2j[4] = c[10]/h2;
	u2j[5] = c[11]/h2;

	pj[0] = c[12]/h;
	pj[1] = c[13]/h2;
	pj[2] = c[14]/h2;

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void 
solve2DStokesCauchyProblem2(const double bdry1_crd[3][2],  // Dirichlet BC nodes
														const double bdry2_crd[2][2], // Neumann BC nodes
														const double bdry2_nml[2][2], // Neumann BC nodes
														const double bulk1_crd[2],  	// PDE1 nodes
														const double bulk2_crd[3][2], // div free nodes
														const double u1[3],						// Dirichlet1 BC val
														const double u2[3],						// Dirichlet2 BC val
														const double Sn1[2],					// Neumann1 BC val
														const double Sn2[2],					// Neumann2 BC val
														double f1, 										// RHS1 val
														double f2, 										// RHS2 val
														const double center[2], 			// patch center
														double mu, 										// viscosity
														double kappa,									// kappa (1/dt)
														double h, 										// grid size
														double u1j[6],								// u1 sol
														double u2j[6],								// u1 sol
														double pj[3])									// p sol
{
	double h2 = h * h;
	//double h2 = 2.0*mu/kappa;
	double kap_h2 = kappa*h2;
	double r_kaph2 = 1.0/kap_h2;

	double mat[15][15], b[15], c[15];

	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = mat[m1][6]  = 1.0;
                
		mat[m][1] = mat[m1][7]  = dx;
		mat[m][2] = mat[m1][8]  = dy;
                
		mat[m][3] = mat[m1][9]  = 0.5*dx*dx;
		mat[m][4] = mat[m1][10] = 0.5*dy*dy;
		mat[m][5] = mat[m1][11] = dx*dy;

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][1] = mu*2.0*nx;
		mat[m][2] = mu*ny;
		mat[m][3] = mu*2.0*nx*dx;
		mat[m][4] = mu*ny*dy;
		mat[m][5] = mu*(2.0*nx*dy + ny*dx);

		mat[m][7] = mu*ny;
		mat[m][9] = mu*ny*dx;
		mat[m][11] = mu*ny*dy;

		mat[m][12] = -nx;
		mat[m][13] = -nx*dx;
		mat[m][14] = -nx*dy;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][2] = mu*nx;
		mat[m][4] = mu*nx*dy;
		mat[m][5] = mu*nx*dx;

		mat[m][7] = mu*nx;
		mat[m][8] = mu*2.0*ny;
		mat[m][9] = mu*nx*dx;
		mat[m][10] = mu*2.0*ny*dy;
		mat[m][11] = mu*(nx*dy+2.0*ny*dx);

		mat[m][12] = -ny;
		mat[m][13] = -ny*dx;
		mat[m][14] = -ny*dy;

		b[m++] = Sn2[l] * h;

	}

	// - mu \Delta u + \kappa u + \nabla p = f

	{
		double x0 = bulk1_crd[0];
		double y0 = bulk1_crd[1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][0]  = kap_h2;
		mat[m][1]  = kap_h2*dx;
		mat[m][2]  = kap_h2*dy;
		mat[m][3]  = kap_h2*0.5*dx*dx - mu;
		mat[m][4]  = kap_h2*0.5*dy*dy - mu;
		mat[m][5]  = kap_h2*dx*dy;
		mat[m][13] = 1.0;

		b[m++] = f1*h2;

		mat[m][6]  = kap_h2;
		mat[m][7]  = kap_h2*dx;
		mat[m][8]  = kap_h2*dy;
		mat[m][9]  = kap_h2*0.5*dx*dx - mu;
		mat[m][10] = kap_h2*0.5*dy*dy - mu;
		mat[m][11] = kap_h2*dx*dy;
		mat[m][14] = 1.0;

		b[m++] = f2*h2;
	}

	// div u = 0

	for(int l = 0; l < 3; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][8] = mat[m][1] = 1.0;
		mat[m][11] = mat[m][3] = dx;
		mat[m][10] = mat[m][5] = dy;

		b[m++] = 0.0;
	}

	assert(m == 15);

	//preConditionSystem<15>(mat, b);
	//bool status = solveByQRdecomposition<15>(mat, b, c, 15);
	bool status = solveByIterativeQR<15>(mat, b, c, 15);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(15, 15);
	//for(int i = 0; i < 15; i++){
	//	for(int j = 0; j < 15; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//}
	//std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;
	//exit(1);

	//for(int i = 3; i < 6; i++){
	//	if (fabs(c[i]) < 1.0e-13) {
	//		c[i] = 0.0;
	//	}
	//}
	//for(int i = 9; i < 12; i++){
	//	if (fabs(c[i]) < 1.0e-13) {
	//		c[i] = 0.0;
	//	}
	//}
	//for(int i = 13; i < 15; i++){
	//	if (fabs(c[i]) < 1.0e-13) {
	//		c[i] = 0.0;
	//	}
	//}

	u1j[0] = c[0];
	u1j[1] = c[1]/h;
	u1j[2] = c[2]/h;
	u1j[3] = c[3]/h2;
	u1j[4] = c[4]/h2;
	u1j[5] = c[5]/h2;

	u2j[0] = c[6];
	u2j[1] = c[7]/h;
	u2j[2] = c[8]/h;
	u2j[3] = c[9]/h2;
	u2j[4] = c[10]/h2;
	u2j[5] = c[11]/h2;

	pj[0] = c[12]/h;
	pj[1] = c[13]/h2;
	pj[2] = c[14]/h2;

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*inline void 
solve2DStokesCauchyProblem2(const double bdry1_crd[3][2],  // Dirichlet BC nodes
														const double bdry2_crd[2][2], // Neumann BC nodes
														const double bdry2_nml[2][2], // Neumann BC nodes
														const double bulk1_crd[2],  	// PDE1 nodes
														const double bulk2_crd[3][2], // div free nodes
														const double u1[3],						// Dirichlet1 BC val
														const double u2[3],						// Dirichlet2 BC val
														const double Sn1[2],					// Neumann1 BC val
														const double Sn2[2],					// Neumann2 BC val
														double f1, 										// RHS1 val
														double f2, 										// RHS2 val
														const double center[2], 			// patch center
														double mu, 										// viscosity
														double kappa,									// kappa (1/dt)
														double h, 										// grid size
														double u1j[6],								// u1 sol
														double u2j[6],								// u1 sol
														double pj[3])									// p sol
{
	double h2 = h * h;
	double kap_h2 = kappa*h2;
	double r_kaph2 = 1.0/kap_h2;

	double mat[15][15], b[15], c[15];

	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]);
		double dy = (y0 - center[1]);

		int m1 = m + 1;

		mat[m][0] = mat[m1][6]  = 1.0;
                
		mat[m][1] = mat[m1][7]  = dx;
		mat[m][2] = mat[m1][8]  = dy;
                
		mat[m][3] = mat[m1][9]  = 0.5*dx*dx;
		mat[m][4] = mat[m1][10] = 0.5*dy*dy;
		mat[m][5] = mat[m1][11] = dx*dy;

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]);
		double dy = (y0 - center[1]);

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][1] = mu*2.0*nx;
		mat[m][2] = mu*ny;
		mat[m][3] = mu*2.0*nx*dx;
		mat[m][4] = mu*ny*dy;
		mat[m][5] = mu*(2.0*nx*dy + ny*dx);

		mat[m][7] = mu*ny;
		mat[m][9] = mu*ny*dx;
		mat[m][11] = mu*ny*dy;

		mat[m][12] = -nx;
		mat[m][13] = -nx*dx;
		mat[m][14] = -nx*dy;

		b[m++] = Sn1[l];

		///////////////////////////////////////////////////////////////////// 

		mat[m][2] = mu*nx;
		mat[m][4] = mu*nx*dy;
		mat[m][5] = mu*nx*dx;

		mat[m][7] = mu*nx;
		mat[m][8] = mu*2.0*ny;
		mat[m][9] = mu*nx*dx;
		mat[m][10] = mu*2.0*ny*dy;
		mat[m][11] = mu*(nx*dy+2.0*ny*dx);

		mat[m][12] = -ny;
		mat[m][13] = -ny*dx;
		mat[m][14] = -ny*dy;

		b[m++] = Sn2[l];

	}

	// - mu \Delta u + \kappa u + \nabla p = f

	{
		double x0 = bulk1_crd[0];
		double y0 = bulk1_crd[1];

		double dx = (x0 - center[0]);
		double dy = (y0 - center[1]);

		mat[m][0]  = kappa;
		mat[m][1]  = kappa*dx;
		mat[m][2]  = kappa*dy;
		mat[m][3]  = kappa*0.5*dx*dx - mu;
		mat[m][4]  = kappa*0.5*dy*dy - mu;
		mat[m][5]  = kappa*dx*dy;
		mat[m][13] = 1.0;

		b[m++] = f1;

		mat[m][6]  = kappa;
		mat[m][7]  = kappa*dx;
		mat[m][8]  = kappa*dy;
		mat[m][9]  = kappa*0.5*dx*dx - mu;
		mat[m][10] = kappa*0.5*dy*dy - mu;
		mat[m][11] = kappa*dx*dy;
		mat[m][14] = 1.0;

		b[m++] = f2;
	}

	// div u = 0

	for(int l = 0; l < 3; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]);
		double dy = (y0 - center[1]);

		mat[m][8] = mat[m][1] = 1.0;
		mat[m][11] = mat[m][3] = dx;
		mat[m][10] = mat[m][5] = dy;

		b[m++] = 0.0;
	}

	assert(m == 15);

	preConditionSystem<15>(mat, b);

	double Lambda[15];

	for(int j = 0; j < 15; j++){
		double maxE = 0.0;
		for(int i = 0; i < 15; i++){
			double tmp = fabs(mat[i][j]);
			maxE = maxE > tmp ? maxE : tmp;
		}
		double r_maxE = 1.0/maxE;
		Lambda[j] = r_maxE;

		for(int i = 0; i < 15; i++){
			mat[i][j] *= r_maxE;
			errorCutoff(mat[i][j]);
		}
	}

	bool status = solveByQRdecomposition<15>(mat, b, c, 15);

	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	Eigen::MatrixXd A(15, 15);
	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			A.coeffRef(i, j) = mat[i][j];
		}
	}
	std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;
	exit(1);

	for(int i = 0; i < 15; i++){
		c[i] *= Lambda[i];
	}

	u1j[0] = c[0];
	u1j[1] = c[1];
	u1j[2] = c[2];
	u1j[3] = c[3];
	u1j[4] = c[4];
	u1j[5] = c[5];

	u2j[0] = c[6];
	u2j[1] = c[7];
	u2j[2] = c[8];
	u2j[3] = c[9];
	u2j[4] = c[10];
	u2j[5] = c[11];

	pj[0] = c[12];
	pj[1] = c[13];
	pj[2] = c[14];

}*/

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*inline void 
solve2DStokesCauchyProblem2(const double bdry1_crd[3][2],  // Dirichlet BC nodes
														const double bdry2_crd[2][2], // Neumann BC nodes
														const double bdry2_nml[2][2], // Neumann BC nodes
														const double bulk1_crd[2],  	// PDE1 nodes
														const double bulk2_crd[3][2], // div free nodes
														const double u1[3],						// Dirichlet1 BC val
														const double u2[3],						// Dirichlet2 BC val
														const double Sn1[2],					// Neumann1 BC val
														const double Sn2[2],					// Neumann2 BC val
														double f1, 										// RHS1 val
														double f2, 										// RHS2 val
														const double center[2], 			// patch center
														double h, 										// grid size
														double u1j[6],								// u1 sol
														double u2j[6],								// u1 sol
														double pj[3])									// p sol
{
	double h2 = h * h;

	double mat[15][15], b[15], c[15];

	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = mat[m1][6]  = 1.0;
                
		mat[m][1] = mat[m1][7]  = dx;
		mat[m][2] = mat[m1][8]  = dy;
                
		mat[m][3] = mat[m1][9]  = 0.5*dx*dx;
		mat[m][4] = mat[m1][10] = 0.5*dy*dy;
		mat[m][5] = mat[m1][11] = dx*dy;

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 2; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		mat[m][1] = 2.0*nx;
		mat[m][2] = ny;
		mat[m][3] = 2.0*nx*dx;
		mat[m][4] = ny*dy;
		mat[m][5] = 2.0*nx*dy + ny*dx;

		mat[m][7] = ny;
		mat[m][9] = ny*dx;
		mat[m][11] = ny*dy;

		mat[m][12] = -h*nx;
		mat[m][13] = -h*nx*dx;
		mat[m][14] = -h*nx*dy;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][2] = nx;
		mat[m][4] = nx*dy;
		mat[m][5] = nx*dx;

		mat[m][7] = nx;
		mat[m][8] = 2.0*ny;
		mat[m][9] = nx*dx;
		mat[m][10] = 2.0*ny*dy;
		mat[m][11] = nx*dy+2.0*ny*dx;

		mat[m][12] = -h*ny;
		mat[m][13] = -h*ny*dx;
		mat[m][14] = -h*ny*dy;

		b[m++] = Sn2[l] * h;

	}

	// - \Delta u + \nabla p = f

	{
		double x0 = bulk1_crd[0];
		double y0 = bulk1_crd[1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][3] = mat[m][4] = -1.0;
		mat[m][13] = h*1.0;

		b[m++] = f1*h2;

		mat[m][9] = mat[m][10] = -1.0;
		mat[m][14] = h*1.0;

		b[m++] = f2*h2;
	}

	// div u = 0

	for(int l = 0; l < 3; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][8] = mat[m][1] = 1.0;
		mat[m][11] = mat[m][3] = dx;
		mat[m][10] = mat[m][5] = dy;

		b[m++] = 0.0;
	}

	assert(m == 15);

	bool status = solveByQRdecomposition<15>(mat, b, c, 15);
	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(15, 15);
	//Eigen::VectorXd sol(15), rhs(15);
	//for(int i = 0; i < 15; i++){
	//	for(int j = 0; j < 15; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//	rhs[i] = b[i];
	//}
	//sol = A.colPivHouseholderQr().solve(rhs);
	//for(int i = 0; i < 15; i++){
	//	c[i] = sol[i];
	//}
	//std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;

	u1j[0] = c[0];
	u1j[1] = c[1]/h;
	u1j[2] = c[2]/h;
	u1j[3] = c[3]/h2;
	u1j[4] = c[4]/h2;
	u1j[5] = c[5]/h2;

	u2j[0] = c[6];
	u2j[1] = c[7]/h;
	u2j[2] = c[8]/h;
	u2j[3] = c[9]/h2;
	u2j[4] = c[10]/h2;
	u2j[5] = c[11]/h2;

	pj[0] = c[12];
	pj[1] = c[13]/h;
	pj[2] = c[14]/h;

}*/

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*inline void 
solve2DStokesCauchyProblem4(const double bdry1_crd[5][2], // Dirichlet BC nodes
														const double bdry2_crd[4][2], // Neumann BC nodes
														const double bdry2_nml[4][2], // Neumann BC nodes
														const double bulk1_crd[6][2], // PDE1 nodes
														const double bulk2_crd[10][2],// div free nodes
														const double u1[5],						// Dirichlet1 BC val
														const double u2[5],						// Dirichlet2 BC val
														const double Sn1[4],					// Neumann1 BC val
														const double Sn2[4],					// Neumann2 BC val
														const double f1[6], 					// RHS1 val
														const double f2[6], 					// RHS2 val
														const double center[2], 			// patch center
														double h, 										// grid size
														double u1j[15],								// u1 sol
														double u2j[15],								// u1 sol
														double pj[10])								// p sol
{
	double h2 = h * h;
	double h3 = h2 * h;
	double h4 = h2 * h2;

	// 15 + 15 + 10 = 40

	double mat[40][40], b[40], c[40];
	for(int i = 0; i < 40; i++){
		for(int j = 0; j < 40; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 5; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = 1.0;
              
		mat[m][1] = dx;
		mat[m][2] = dy;
              
		mat[m][3] = 0.5*dx*dx;
		mat[m][4] = 0.5*dy*dy;
		mat[m][5] = dx*dy;

		mat[m][6] = dx*dx*dx/6.0;
		mat[m][7] = dy*dy*dy/6.0;
		mat[m][8] = 0.5*dx*dx*dy;
		mat[m][9] = 0.5*dy*dy*dx;

		mat[m][10] = dx*dx*dx*dx/24.0;
		mat[m][11] = dy*dy*dy*dy/24.0;
		mat[m][12] = dx*dx*dx*dy/6.0;
		mat[m][13] = dy*dy*dy*dx/6.0;
		mat[m][14] = 0.25*dx*dx*dy*dy;

		for(int j = 0, j1 = 15; j < 15; j++, j1++){
			mat[m1][j1] = mat[m][j];
		}

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 4; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		// (2u1_x - p) * nx + (u1_y + u2_x) * uy

		mat[m][1]  = 2.0*nx *1.0;
		mat[m][3]  = 2.0*nx *dx;
		mat[m][5]  = 2.0*nx *dy;
		mat[m][6]  = 2.0*nx *0.5*dx*dx;
		mat[m][8]  = 2.0*nx *dx*dy;
		mat[m][9]  = 2.0*nx *0.5*dy*dy;
		mat[m][10] = 2.0*nx *dx*dx*dx/6.0;
		mat[m][12] = 2.0*nx *0.5*dx*dx*dy;
		mat[m][13] = 2.0*nx *dy*dy*dy/6.0;
		mat[m][14] = 2.0*nx *0.5*dx*dy*dy;

		mat[m][2]  += ny *1.0;
		mat[m][4]  += ny *dy;
		mat[m][5]  += ny *dx;
		mat[m][7]  += ny *0.5*dy*dy;
		mat[m][8]  += ny *0.5*dx*dx;
		mat[m][9]  += ny *dy*dx;
		mat[m][11] += ny *dy*dy*dy/6.0;
		mat[m][12] += ny *dx*dx*dx/6.0;
		mat[m][13] += ny *0.5*dx*dy*dy;
		mat[m][14] += ny *0.5*dx*dx*dy;

		mat[m][15+1]  = ny *1.0;
		mat[m][15+3]  = ny *dx;
		mat[m][15+5]  = ny *dy;
		mat[m][15+6]  = ny *0.5*dx*dx;
		mat[m][15+8]  = ny *dx*dy;
		mat[m][15+9]  = ny *0.5*dy*dy;
		mat[m][15+10] = ny *dx*dx*dx/6.0;
		mat[m][15+12] = ny *0.5*dx*dx*dy;
		mat[m][15+13] = ny *dy*dy*dy/6.0;
		mat[m][15+14] = ny *0.5*dx*dy*dy;

		mat[m][30+0] = -nx *1.0;
		mat[m][30+1] = -nx *dx;
		mat[m][30+2] = -nx *dy;
		mat[m][30+3] = -nx *0.5*dx*dx;
		mat[m][30+4] = -nx *0.5*dy*dy;
		mat[m][30+5] = -nx *dx*dy;
		mat[m][30+6] = -nx *dx*dx*dx/6.0;
		mat[m][30+7] = -nx *dy*dy*dy/6.0;
		mat[m][30+8] = -nx *0.5*dx*dx*dy;
		mat[m][30+9] = -nx *0.5*dy*dy*dx;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 
		// (u1_y + u2_x) * ux + (2u2_y - p) * ny

		mat[m][2]  = nx *1.0;
		mat[m][4]  = nx *dy;
		mat[m][5]  = nx *dx;
		mat[m][7]  = nx *0.5*dy*dy;
		mat[m][8]  = nx *0.5*dx*dx;
		mat[m][9]  = nx *dy*dx;
		mat[m][11] = nx *dy*dy*dy/6.0;
		mat[m][12] = nx *dx*dx*dx/6.0;
		mat[m][13] = nx *0.5*dx*dy*dy;
		mat[m][14] = nx *0.5*dx*dx*dy;

		mat[m][15+2]  = 2.0*ny *1.0;
		mat[m][15+4]  = 2.0*ny *dy;
		mat[m][15+5]  = 2.0*ny *dx;
		mat[m][15+7]  = 2.0*ny *0.5*dy*dy;
		mat[m][15+8]  = 2.0*ny *0.5*dx*dx;
		mat[m][15+9]  = 2.0*ny *dy*dx;
		mat[m][15+11] = 2.0*ny *dy*dy*dy/6.0;
		mat[m][15+12] = 2.0*ny *dx*dx*dx/6.0;
		mat[m][15+13] = 2.0*ny *0.5*dx*dy*dy;
		mat[m][15+14] = 2.0*ny *0.5*dx*dx*dy;

		mat[m][15+1]  += nx *1.0;
		mat[m][15+3]  += nx *dx;
		mat[m][15+5]  += nx *dy;
		mat[m][15+6]  += nx *0.5*dx*dx;
		mat[m][15+8]  += nx *dx*dy;
		mat[m][15+9]  += nx *0.5*dy*dy;
		mat[m][15+10] += nx *dx*dx*dx/6.0;
		mat[m][15+12] += nx *0.5*dx*dx*dy;
		mat[m][15+13] += nx *dy*dy*dy/6.0;
		mat[m][15+14] += nx *0.5*dx*dy*dy;

		mat[m][30+0] = -ny *1.0;
		mat[m][30+1] = -ny *dx;
		mat[m][30+2] = -ny *dy;
		mat[m][30+3] = -ny *0.5*dx*dx;
		mat[m][30+4] = -ny *0.5*dy*dy;
		mat[m][30+5] = -ny *dx*dy;
		mat[m][30+6] = -ny *dx*dx*dx/6.0;
		mat[m][30+7] = -ny *dy*dy*dy/6.0;
		mat[m][30+8] = -ny *0.5*dx*dx*dy;
		mat[m][30+9] = -ny *0.5*dy*dy*dx;

		b[m++] = Sn2[l] * h;

	}

	// - \Delta u + \nabla p = f

	for(int l = 0; l < 6; l++){

		double x0 = bulk1_crd[l][0];
		double y0 = bulk1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][3] = -1.0;
		mat[m][4] = -1.0;
		mat[m][6] = -dx;
		mat[m][7] = -dy;
		mat[m][8] = -dy;
		mat[m][9] = -dx;
		mat[m][10] = -0.5*dx*dx;
		mat[m][11] = -0.5*dy*dy;
		mat[m][12] = -dx*dy;
		mat[m][13] = -dy*dx;
		mat[m][14] = -0.5*(dy*dy+dx*dx);

		mat[m][30+1] = 1.0;
		mat[m][30+3] = dx;
		mat[m][30+5] = dy;
		mat[m][30+6] = 0.5*dx*dx;
		mat[m][30+8] = dx*dy;
		mat[m][30+9] = 0.5*dy*dy;

		b[m++] = f1[l]*h2;

		mat[m][15+3]  = -1.0;
		mat[m][15+4]  = -1.0;
		mat[m][15+6]  = -dx;
		mat[m][15+7]  = -dy;
		mat[m][15+8]  = -dy;
		mat[m][15+9]  = -dx;
		mat[m][15+10] = -0.5*dx*dx;
		mat[m][15+11] = -0.5*dy*dy;
		mat[m][15+12] = -dx*dy;
		mat[m][15+13] = -dy*dx;
		mat[m][15+14] = -0.5*(dy*dy+dx*dx);

		mat[m][30+2] = 1.0;
		mat[m][30+4] = dy;
		mat[m][30+5] = dx;
		mat[m][30+7] = 0.5*dy*dy;
		mat[m][30+8] = 0.5*dx*dx;
		mat[m][30+9] = dy*dx;

		b[m++] = f2[l]*h2;
	}

	// div u = 0

	for(int l = 0; l < 10; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][1]  = 1.0;
		mat[m][3]  = dx;
		mat[m][5]  = dy;
		mat[m][6]  = 0.5*dx*dx;
		mat[m][8]  = dx*dy;
		mat[m][9]  = 0.5*dy*dy;
		mat[m][10] = dx*dx*dx/6.0;
		mat[m][12] = 0.5*dx*dx*dy;
		mat[m][13] = dy*dy*dy/6.0;
		mat[m][14] = 0.5*dx*dy*dy;

		mat[m][15+2]  += 1.0;
		mat[m][15+4]  += dy;
		mat[m][15+5]  += dx;
		mat[m][15+7]  += 0.5*dy*dy;
		mat[m][15+8]  += 0.5*dx*dx;
		mat[m][15+9]  += dy*dx;
		mat[m][15+11] += dy*dy*dy/6.0;
		mat[m][15+12] += dx*dx*dx/6.0;
		mat[m][15+13] += 0.5*dx*dy*dy;
		mat[m][15+14] += 0.5*dx*dx*dy;

		b[m++] = 0.0;
	}

	//bool status = solveByQRdecomposition<40>(mat, b, c, 40);
	//if (!status) {
	//	std::cout << "failed to solve by QR decomposition." << std::endl;
	//	exit(1);
	//}

	Eigen::MatrixXd A(40, 40);
	Eigen::VectorXd sol(40), rhs(40);
	for(int i = 0; i < 40; i++){
		for(int j = 0; j < 40; j++){
			A.coeffRef(i, j) = mat[i][j];
		}
		rhs[i] = b[i];
	}
	sol = A.colPivHouseholderQr().solve(rhs);
	for(int i = 0; i < 40; i++){
		c[i] = sol[i];
	}
	std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;

	u1j[0] = c[0];
	u2j[0] = c[15];
	pj[0] = c[30];

	for(int l = 1; l < 3; l++){
		u1j[l] = c[l]/h;
		u2j[l] = c[l+15]/h;
		pj[l] = c[l+30]/h2;
	}
	for(int l = 3; l < 6; l++){
		u1j[l] = c[l]/h2;
		u2j[l] = c[l+15]/h2;
		pj[l] = c[l+30]/h3;
	}
	for(int l = 6; l < 10; l++){
		u1j[l] = c[l]/h3;
		u2j[l] = c[l+15]/h3;
		pj[l] = c[l+30]/h4;
	}
	for(int l = 10; l < 15; l++){
		u1j[l] = c[l]/h4;
		u2j[l] = c[l+15]/h4;
	}


}*/

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void 
solve2DStokesCauchyProblem4(const double bdry1_crd[5][2], // Dirichlet BC nodes
														const double bdry2_crd[4][2], // Neumann BC nodes
														const double bdry2_nml[4][2], // Neumann BC nodes
														const double bulk1_crd[6][2], // PDE1 nodes
														const double bulk2_crd[10][2],// div free nodes
														const double u1[5],						// Dirichlet1 BC val
														const double u2[5],						// Dirichlet2 BC val
														const double Sn1[4],					// Neumann1 BC val
														const double Sn2[4],					// Neumann2 BC val
														const double f1[6], 					// RHS1 val
														const double f2[6], 					// RHS2 val
														const double center[2], 			// patch center
														double h, 										// grid size
														double u1j[15],								// u1 sol
														double u2j[15],								// u1 sol
														double pj[10])								// p sol
{
	double h2 = h * h;
	double h3 = h2 * h;
	double h4 = h2 * h2;

	// 15 + 15 + 10 = 40

	double mat[40][40], b[40], c[40];
	for(int i = 0; i < 40; i++){
		for(int j = 0; j < 40; j++){
			mat[i][j] = 0.0;
		}
		b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 5; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		int m1 = m + 1;

		mat[m][0] = 1.0;
              
		mat[m][1] = dx;
		mat[m][2] = dy;
              
		mat[m][3] = 0.5*dx*dx;
		mat[m][4] = 0.5*dy*dy;
		mat[m][5] = dx*dy;

		mat[m][6] = dx*dx*dx/6.0;
		mat[m][7] = dy*dy*dy/6.0;
		mat[m][8] = 0.5*dx*dx*dy;
		mat[m][9] = 0.5*dy*dy*dx;

		mat[m][10] = dx*dx*dx*dx/24.0;
		mat[m][11] = dy*dy*dy*dy/24.0;
		mat[m][12] = dx*dx*dx*dy/6.0;
		mat[m][13] = dy*dy*dy*dx/6.0;
		mat[m][14] = 0.25*dx*dx*dy*dy;

		for(int j = 0, j1 = 15; j < 15; j++, j1++){
			mat[m1][j1] = mat[m][j];
		}

		b[m] = u1[l];
		b[m1] = u2[l];

		m += 2;
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 4; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];

		// (2u1_x - p) * nx + (u1_y + u2_x) * uy

		mat[m][1]  = 2.0*nx *1.0;
		mat[m][3]  = 2.0*nx *dx;
		mat[m][5]  = 2.0*nx *dy;
		mat[m][6]  = 2.0*nx *0.5*dx*dx;
		mat[m][8]  = 2.0*nx *dx*dy;
		mat[m][9]  = 2.0*nx *0.5*dy*dy;
		mat[m][10] = 2.0*nx *dx*dx*dx/6.0;
		mat[m][12] = 2.0*nx *0.5*dx*dx*dy;
		mat[m][13] = 2.0*nx *dy*dy*dy/6.0;
		mat[m][14] = 2.0*nx *0.5*dx*dy*dy;

		mat[m][2]  += ny *1.0;
		mat[m][4]  += ny *dy;
		mat[m][5]  += ny *dx;
		mat[m][7]  += ny *0.5*dy*dy;
		mat[m][8]  += ny *0.5*dx*dx;
		mat[m][9]  += ny *dy*dx;
		mat[m][11] += ny *dy*dy*dy/6.0;
		mat[m][12] += ny *dx*dx*dx/6.0;
		mat[m][13] += ny *0.5*dx*dy*dy;
		mat[m][14] += ny *0.5*dx*dx*dy;

		mat[m][15+1]  = ny *1.0;
		mat[m][15+3]  = ny *dx;
		mat[m][15+5]  = ny *dy;
		mat[m][15+6]  = ny *0.5*dx*dx;
		mat[m][15+8]  = ny *dx*dy;
		mat[m][15+9]  = ny *0.5*dy*dy;
		mat[m][15+10] = ny *dx*dx*dx/6.0;
		mat[m][15+12] = ny *0.5*dx*dx*dy;
		mat[m][15+13] = ny *dy*dy*dy/6.0;
		mat[m][15+14] = ny *0.5*dx*dy*dy;

		mat[m][30+0] = -h*nx *1.0;
		mat[m][30+1] = -h*nx *dx;
		mat[m][30+2] = -h*nx *dy;
		mat[m][30+3] = -h*nx *0.5*dx*dx;
		mat[m][30+4] = -h*nx *0.5*dy*dy;
		mat[m][30+5] = -h*nx *dx*dy;
		mat[m][30+6] = -h*nx *dx*dx*dx/6.0;
		mat[m][30+7] = -h*nx *dy*dy*dy/6.0;
		mat[m][30+8] = -h*nx *0.5*dx*dx*dy;
		mat[m][30+9] = -h*nx *0.5*dy*dy*dx;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 
		// (u1_y + u2_x) * ux + (2u2_y - p) * ny

		mat[m][2]  = nx *1.0;
		mat[m][4]  = nx *dy;
		mat[m][5]  = nx *dx;
		mat[m][7]  = nx *0.5*dy*dy;
		mat[m][8]  = nx *0.5*dx*dx;
		mat[m][9]  = nx *dy*dx;
		mat[m][11] = nx *dy*dy*dy/6.0;
		mat[m][12] = nx *dx*dx*dx/6.0;
		mat[m][13] = nx *0.5*dx*dy*dy;
		mat[m][14] = nx *0.5*dx*dx*dy;

		mat[m][15+2]  = 2.0*ny *1.0;
		mat[m][15+4]  = 2.0*ny *dy;
		mat[m][15+5]  = 2.0*ny *dx;
		mat[m][15+7]  = 2.0*ny *0.5*dy*dy;
		mat[m][15+8]  = 2.0*ny *0.5*dx*dx;
		mat[m][15+9]  = 2.0*ny *dy*dx;
		mat[m][15+11] = 2.0*ny *dy*dy*dy/6.0;
		mat[m][15+12] = 2.0*ny *dx*dx*dx/6.0;
		mat[m][15+13] = 2.0*ny *0.5*dx*dy*dy;
		mat[m][15+14] = 2.0*ny *0.5*dx*dx*dy;

		mat[m][15+1]  += nx *1.0;
		mat[m][15+3]  += nx *dx;
		mat[m][15+5]  += nx *dy;
		mat[m][15+6]  += nx *0.5*dx*dx;
		mat[m][15+8]  += nx *dx*dy;
		mat[m][15+9]  += nx *0.5*dy*dy;
		mat[m][15+10] += nx *dx*dx*dx/6.0;
		mat[m][15+12] += nx *0.5*dx*dx*dy;
		mat[m][15+13] += nx *dy*dy*dy/6.0;
		mat[m][15+14] += nx *0.5*dx*dy*dy;

		mat[m][30+0] = -h*ny *1.0;
		mat[m][30+1] = -h*ny *dx;
		mat[m][30+2] = -h*ny *dy;
		mat[m][30+3] = -h*ny *0.5*dx*dx;
		mat[m][30+4] = -h*ny *0.5*dy*dy;
		mat[m][30+5] = -h*ny *dx*dy;
		mat[m][30+6] = -h*ny *dx*dx*dx/6.0;
		mat[m][30+7] = -h*ny *dy*dy*dy/6.0;
		mat[m][30+8] = -h*ny *0.5*dx*dx*dy;
		mat[m][30+9] = -h*ny *0.5*dy*dy*dx;

		b[m++] = Sn2[l] * h;

	}

	// - \Delta u + \nabla p = f

	for(int l = 0; l < 6; l++){

		double x0 = bulk1_crd[l][0];
		double y0 = bulk1_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][3] = -1.0;
		mat[m][4] = -1.0;
		mat[m][6] = -dx;
		mat[m][7] = -dy;
		mat[m][8] = -dy;
		mat[m][9] = -dx;
		mat[m][10] = -0.5*dx*dx;
		mat[m][11] = -0.5*dy*dy;
		mat[m][12] = -dx*dy;
		mat[m][13] = -dy*dx;
		mat[m][14] = -0.5*(dy*dy+dx*dx);

		mat[m][30+1] = h*1.0;
		mat[m][30+3] = h*dx;
		mat[m][30+5] = h*dy;
		mat[m][30+6] = h*0.5*dx*dx;
		mat[m][30+8] = h*dx*dy;
		mat[m][30+9] = h*0.5*dy*dy;

		b[m++] = f1[l]*h2;

		mat[m][15+3]  = -1.0;
		mat[m][15+4]  = -1.0;
		mat[m][15+6]  = -dx;
		mat[m][15+7]  = -dy;
		mat[m][15+8]  = -dy;
		mat[m][15+9]  = -dx;
		mat[m][15+10] = -0.5*dx*dx;
		mat[m][15+11] = -0.5*dy*dy;
		mat[m][15+12] = -dx*dy;
		mat[m][15+13] = -dy*dx;
		mat[m][15+14] = -0.5*(dy*dy+dx*dx);

		mat[m][30+2] = h*1.0;
		mat[m][30+4] = h*dy;
		mat[m][30+5] = h*dx;
		mat[m][30+7] = h*0.5*dy*dy;
		mat[m][30+8] = h*0.5*dx*dx;
		mat[m][30+9] = h*dy*dx;

		b[m++] = f2[l]*h2;
	}

	// div u = 0

	for(int l = 0; l < 10; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;

		mat[m][1]  = 1.0;
		mat[m][3]  = dx;
		mat[m][5]  = dy;
		mat[m][6]  = 0.5*dx*dx;
		mat[m][8]  = dx*dy;
		mat[m][9]  = 0.5*dy*dy;
		mat[m][10] = dx*dx*dx/6.0;
		mat[m][12] = 0.5*dx*dx*dy;
		mat[m][13] = dy*dy*dy/6.0;
		mat[m][14] = 0.5*dx*dy*dy;

		mat[m][15+2]  += 1.0;
		mat[m][15+4]  += dy;
		mat[m][15+5]  += dx;
		mat[m][15+7]  += 0.5*dy*dy;
		mat[m][15+8]  += 0.5*dx*dx;
		mat[m][15+9]  += dy*dx;
		mat[m][15+11] += dy*dy*dy/6.0;
		mat[m][15+12] += dx*dx*dx/6.0;
		mat[m][15+13] += 0.5*dx*dy*dy;
		mat[m][15+14] += 0.5*dx*dx*dy;

		b[m++] = 0.0;
	}

	bool status = solveByQRdecomposition<40>(mat, b, c, 40);
	if (!status) {
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(40, 40);
	//Eigen::VectorXd sol(40), rhs(40);
	//for(int i = 0; i < 40; i++){
	//	for(int j = 0; j < 40; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//	rhs[i] = b[i];
	//}
	//sol = A.colPivHouseholderQr().solve(rhs);
	//for(int i = 0; i < 40; i++){
	//	c[i] = sol[i];
	//}

	u1j[0] = c[0];
	u2j[0] = c[15];
	pj[0] = c[30];

	for(int l = 1; l < 3; l++){
		u1j[l] = c[l]/h;
		u2j[l] = c[l+15]/h;
		pj[l] = c[l+30]/h;
	}
	for(int l = 3; l < 6; l++){
		u1j[l] = c[l]/h2;
		u2j[l] = c[l+15]/h2;
		pj[l] = c[l+30]/h2;
	}
	for(int l = 6; l < 10; l++){
		u1j[l] = c[l]/h3;
		u2j[l] = c[l+15]/h3;
		pj[l] = c[l+30]/h3;
	}
	for(int l = 10; l < 15; l++){
		u1j[l] = c[l]/h4;
		u2j[l] = c[l+15]/h4;
	}

}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline void 
solve3DStokesCauchyProblem2(const double bdry1_crd[6][3], // Dirichlet BC nodes
														const double bdry2_crd[3][3], // Neumann BC nodes
														const double bdry2_nml[3][3], // Neumann BC nodes
														const double bulk1_crd[3],  	// PDE1 nodes
														const double bulk2_crd[4][3], // div free nodes
														const double u1[6],						// Dirichlet1 BC val
														const double u2[6],						// Dirichlet2 BC val
														const double u3[6],						// Dirichlet3 BC val
														const double Sn1[3],					// Neumann1 BC val
														const double Sn2[3],					// Neumann2 BC val
														const double Sn3[3],					// Neumann2 BC val
														double f1, 								    // RHS1 val
														double f2, 								    // RHS2 val
														double f3, 								    // RHS3 val
														double g[4], 								  // div RHS val
														const double center[3], 			// patch center
														double kappa,									// kappa (1/dt)
														double h, 										// grid size
														double u1j[10],								// u1 sol
														double u2j[10],								// u2 sol
														double u3j[10],								// u3 sol
														double pj[4])									// p sol
{

	double h2 = h * h;
	double kap_h2 = kappa*h2;
	errorCutoff(kap_h2);

	double mat[34][34], b[34], c[34];
	for(int i = 0; i < 34; i++){
		for(int j = 0; j < 34; j++){
			mat[i][j] = 0.0;
		}
		c[i] = b[i] = 0.0;
	}

	int m = 0;

	// [u1] = phi1, [u2] = phi2

	for(int l = 0; l < 6; l++){

		double x0 = bdry1_crd[l][0];
		double y0 = bdry1_crd[l][1];
		double z0 = bdry1_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		int m1 = m+1;
		int m2 = m+2;

		mat[m][0] = mat[m1][10] = mat[m2][20] = 1.0;
		mat[m][1] = mat[m1][11] = mat[m2][21] = dx;
		mat[m][2] = mat[m1][12] = mat[m2][22] = dy;
		mat[m][3] = mat[m1][13] = mat[m2][23] = dz;
		mat[m][4] = mat[m1][14] = mat[m2][24] = 0.5*dx*dx;
		mat[m][5] = mat[m1][15] = mat[m2][25] = 0.5*dy*dy;
		mat[m][6] = mat[m1][16] = mat[m2][26] = 0.5*dz*dz;
		mat[m][7] = mat[m1][17] = mat[m2][27] = dy*dz;
		mat[m][8] = mat[m1][18] = mat[m2][28] = dz*dx;
		mat[m][9] = mat[m1][19] = mat[m2][29] = dx*dy;

		b[m++] = u1[l];
		b[m++] = u2[l];
		b[m++] = u3[l];
	}

	// [S1n] = psi1, [S2n] = psi2

	for(int l = 0; l < 3; l++){

		double x0 = bdry2_crd[l][0];
		double y0 = bdry2_crd[l][1];
		double z0 = bdry2_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		double nx = bdry2_nml[l][0];
		double ny = bdry2_nml[l][1];
		double nz = bdry2_nml[l][2];

		mat[m][1]  += 2.0*nx;
		mat[m][4]  += 2.0*nx*dx;
		mat[m][8]  += 2.0*nx*dz;
		mat[m][9]  += 2.0*nx*dy;
		mat[m][30] += -nx;
		mat[m][31] += -nx*dx;
		mat[m][32] += -nx*dy;
		mat[m][33] += -nx*dz;

		mat[m][2]  += ny;
		mat[m][5]  += ny*dy;
		mat[m][7]  += ny*dz;
		mat[m][9]  += ny*dx;
		mat[m][11] += ny;
		mat[m][14] += ny*dx;
		mat[m][18] += ny*dz;
		mat[m][19] += ny*dy;

		mat[m][3]  += nz;
		mat[m][6]  += nz*dz;
		mat[m][7]  += nz*dy;
		mat[m][8]  += nz*dx;
		mat[m][21] += nz;
		mat[m][24] += nz*dx;
		mat[m][28] += nz*dz;
		mat[m][29] += nz*dy;

		b[m++] = Sn1[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][2]  += nx;
		mat[m][5]  += nx*dy;
		mat[m][7]  += nx*dz;
		mat[m][9]  += nx*dx;
		mat[m][11] += nx;
		mat[m][14] += nx*dx;
		mat[m][18] += nx*dz;
		mat[m][19] += nx*dy;

		mat[m][12] += 2.0*ny;
		mat[m][15] += 2.0*ny*dy;
		mat[m][17] += 2.0*ny*dz;
		mat[m][19] += 2.0*ny*dx;
		mat[m][30] += -ny;
		mat[m][31] += -ny*dx;
		mat[m][32] += -ny*dy;
		mat[m][33] += -ny*dz;

		mat[m][22] += nz;
		mat[m][25] += nz*dy;
		mat[m][27] += nz*dz;
		mat[m][29] += nz*dx;
		mat[m][13] += nz;
		mat[m][16] += nz*dz;
		mat[m][17] += nz*dy;
		mat[m][18] += nz*dx;

		b[m++] = Sn2[l] * h;

		///////////////////////////////////////////////////////////////////// 

		mat[m][3]  += nx;
		mat[m][6]  += nx*dz;
		mat[m][7]  += nx*dy;
		mat[m][8]  += nx*dx;
		mat[m][21] += nx;
		mat[m][24] += nx*dx;
		mat[m][28] += nx*dz;
		mat[m][29] += nx*dy;

		mat[m][13] += ny;
		mat[m][16] += ny*dz;
		mat[m][17] += ny*dy;
		mat[m][18] += ny*dx;
		mat[m][22] += ny;
		mat[m][25] += ny*dy;
		mat[m][27] += ny*dz;
		mat[m][29] += ny*dx;

		mat[m][23] += 2.0*nz;
		mat[m][26] += 2.0*nz*dz;
		mat[m][27] += 2.0*nz*dy;
		mat[m][28] += 2.0*nz*dx;
		mat[m][30] += -nz;
		mat[m][31] += -nz*dx;
		mat[m][32] += -nz*dy;
		mat[m][33] += -nz*dz;

		b[m++] = Sn3[l] * h;

	}

	// - \Delta u + \kappa u + \nabla p = f

	{
		double x0 = bulk1_crd[0];
		double y0 = bulk1_crd[1];
		double z0 = bulk1_crd[2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		int m1 = m+1;
		int m2 = m+2;

		mat[m][0]  = mat[m1][10] = mat[m2][20] = kap_h2;
		mat[m][1]  = mat[m1][11] = mat[m2][21] = kap_h2*dx;
		mat[m][2]  = mat[m1][12] = mat[m2][22] = kap_h2*dy;
		mat[m][3]  = mat[m1][13] = mat[m2][23] = kap_h2*dz;
		mat[m][4]  = mat[m1][14] = mat[m2][24] = kap_h2*0.5*dx*dx - 1.0;
		mat[m][5]  = mat[m1][15] = mat[m2][25] = kap_h2*0.5*dy*dy - 1.0;
		mat[m][6]  = mat[m1][16] = mat[m2][26] = kap_h2*0.5*dz*dz - 1.0;
		mat[m][7]  = mat[m1][17] = mat[m2][27] = kap_h2*dy*dz;
		mat[m][8]  = mat[m1][18] = mat[m2][28] = kap_h2*dz*dx;
		mat[m][9]  = mat[m1][19] = mat[m2][29] = kap_h2*dx*dy;
		mat[m][31] = mat[m1][32] = mat[m2][33] = 1.0;

		b[m++] = f1*h2;
		b[m++] = f2*h2;
		b[m++] = f3*h2;
	}

	// - div u = g

	for(int l = 0; l < 4; l++){

		double x0 = bulk2_crd[l][0];
		double y0 = bulk2_crd[l][1];
		double z0 = bulk2_crd[l][2];

		double dx = (x0 - center[0]) / h;
		double dy = (y0 - center[1]) / h;
		double dz = (z0 - center[2]) / h;

		mat[m][1] =	mat[m][12] = mat[m][23] = -1.0;
		mat[m][4] =	mat[m][19] = mat[m][28] = -dx;
		mat[m][9] =	mat[m][15] = mat[m][27] = -dy;
		mat[m][8] =	mat[m][17] = mat[m][26] = -dz;

		b[m++] = g[l] * h;
	}

	assert(m == 34);

	preConditionSystem<34>(mat, b);

	bool status = solveByQRdecomposition<34>(mat, b, c, 34);
	if (!status) {
		for(int m = 0; m < 6; m++){
			std::cout << bdry1_crd[m][0] << ", " << bdry1_crd[m][1] << ", " 
								<< bdry1_crd[m][2] << std::endl;
		}
		std::cout << "failed to solve by QR decomposition." << std::endl;
		exit(1);
	}

	//Eigen::MatrixXd A(34, 34);
	//Eigen::VectorXd sol(34), rhs(34);
	//for(int i = 0; i < 34; i++){
	//	for(int j = 0; j < 34; j++){
	//		A.coeffRef(i, j) = mat[i][j];
	//	}
	//	rhs[i] = b[i];
	//}
	//sol = A.colPivHouseholderQr().solve(rhs);
	//for(int i = 0; i < 34; i++){
	//	c[i] = sol[i];
	//}
	//std::cout << "cond = " << A.inverse().norm() * A.norm() << std::endl;
	//exit(1);

	for(int i = 0; i < 34; i++){
		errorCutoff(c[i]);
	}
	//errorCutoff(c[0]);
	//errorCutoff(c[10]);
	//errorCutoff(c[20]);
	//for(int i = 1, i1 = 11, i2 = 21; i < 4; i++, i1++, i2++){
	//	if (fabs(c[i]) < 1.0e-14) {c[i] = 0.0;};
	//	if (fabs(c[i1]) < 1.0e-14) {c[i1] = 0.0;};
	//	if (fabs(c[i2]) < 1.0e-14) {c[i2] = 0.0;};
	//}
	//for(int i = 4, i1 = 14, i2 = 24; i < 10; i++, i1++, i2++){
	//	if (fabs(c[i]) < 1.0e-13) {c[i] = 0.0;};
	//	if (fabs(c[i1]) < 1.0e-13) {c[i1] = 0.0;};
	//	if (fabs(c[i2]) < 1.0e-13) {c[i2] = 0.0;};
	//}
	//if (fabs(c[30]) < 1.0e-14) {c[30] = 0.0;};
	//for(int i = 31; i < 34; i++){
	//	if (fabs(c[i]) < 1.0e-13) {c[i] = 0.0;};
	//}

	u1j[0] = c[0];
	u2j[0] = c[10];
	u3j[0] = c[20];
	for(int l = 1; l < 4; l++){
		u1j[l] = c[l]/h;
		u2j[l] = c[10+l]/h;
		u3j[l] = c[20+l]/h;
	}
	for(int l = 4; l < 10; l++){
		u1j[l] = c[l]/h2;
		u2j[l] = c[10+l]/h2;
		u3j[l] = c[20+l]/h2;
	}

	pj[0] = c[30]/h;
	pj[1] = c[31]/h2;
	pj[2] = c[32]/h2;
	pj[3] = c[33]/h2;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate1d2(double x, const double c[2])
{
	return c[0]+x*c[1];
}
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate1d3(double x, const double c[3])
{
	return c[0]+x*(c[1]+x*c[2]/2.0);
}
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate1d4(double x, const double c[4])
{
	return c[0]+x*(c[1]+x*(c[2]/2.0+x*c[3]/6.0));
}
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate1d5(double x, const double c[5])
{
	return c[0]+x*(c[1]+x*(c[2]/2.0+x*(c[3]/6.0+x*c[4]/24.0)));
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d3(double x, double y, const double c[3])
{
	double val[6];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	double u = 0.0;
	for(int k = 0; k < 3; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d6(double x, double y, const double c[6])
{
	double val[6];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	double u = 0.0;
	for(int k = 0; k < 6; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d10(double x, double y, const double c[10])
{
	double val[10];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	val[6] = x*x*x/6.0;
	val[7] = y*y*y/6.0;
	val[8] = 0.5*x*x*y;
	val[9] = 0.5*y*y*x;

	double u = 0.0;
	for(int k = 0; k < 10; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d15(double x, double y, const double c[15])
{
	double val[15];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	val[6] = x*x*x/6.0;
	val[7] = y*y*y/6.0;
	val[8] = 0.5*x*x*y;
	val[9] = 0.5*y*y*x;

	val[10] = x*x*x*x/24.0;
	val[11] = y*y*y*y/24.0;
	val[12] = x*x*x*y/6.0;
	val[13] = y*y*y*x/6.0;
	val[14] = 0.25*x*x*y*y;

	double u = 0.0;
	for(int k = 0; k < 15; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d3(double x, double y, const Array<double,3> &c)
{
	double val[6];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	double u = 0.0;
	for(int k = 0; k < 3; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d6(double x, double y, const Array<double,6> &c)
{
	double val[6];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	double u = 0.0;
	for(int k = 0; k < 6; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d10(double x, double y, const Array<double,10> &c)
{
	double val[10];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	val[6] = x*x*x/6.0;
	val[7] = y*y*y/6.0;
	val[8] = 0.5*x*x*y;
	val[9] = 0.5*y*y*x;

	double u = 0.0;
	for(int k = 0; k < 10; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate2d15(double x, double y, const Array<double,15> &c)
{
	double val[15];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;

	val[3] = 0.5*x*x;
	val[4] = 0.5*y*y;
	val[5] = x*y;

	val[6] = x*x*x/6.0;
	val[7] = y*y*y/6.0;
	val[8] = 0.5*x*x*y;
	val[9] = 0.5*y*y*x;

	val[10] = x*x*x*x/24.0;
	val[11] = y*y*y*y/24.0;
	val[12] = x*x*x*y/6.0;
	val[13] = y*y*y*x/6.0;
	val[14] = 0.25*x*x*y*y;

	double u = 0.0;
	for(int k = 0; k < 15; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate3d4(double x, double y, double z, const double c[4])
{
	double val[4];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;
	val[3] = z;

	double u = 0.0;
	for(int k = 0; k < 4; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate3d10(double x, double y, double z, const double c[10])
{
	double val[10];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;
	val[3] = z;

	val[4] = 0.5*x*x;
	val[5] = 0.5*y*y;
	val[6] = 0.5*z*z;
	val[7] = y*z;
	val[8] = z*x;
	val[9] = x*y;

	double u = 0.0;
	for(int k = 0; k < 10; k++){
		u += c[k] * val[k];
	}

	return u;
}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate3d20(double x, double y, double z, const double c[20])
{
	double val[20];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;
	val[3] = z;

	val[4] = 0.5*x*x;
	val[5] = 0.5*y*y;
	val[6] = 0.5*z*z;
	val[7] = y*z;
	val[8] = z*x;
	val[9] = x*y;

	val[10] = x*x*x/6.0;
	val[11] = y*y*y/6.0;
	val[12] = z*z*z/6.0;
	val[13] = 0.5*x*x*y;
	val[14] = 0.5*x*x*z;
	val[15] = 0.5*y*y*x;
	val[16] = 0.5*y*y*z;
	val[17] = 0.5*z*z*x;
	val[18] = 0.5*z*z*y;
	val[19] = x*y*z;

	double u = 0.0;
	for(int k = 0; k < 20; k++){
		u += c[k] * val[k];
	}

	return u;
}


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
inline double evaluate3d35(double x, double y, double z, const double c[35])
{
	double val[35];

	val[0] = 1.0;

	val[1] = x;
	val[2] = y;
	val[3] = z;

	val[4] = 0.5*x*x;
	val[5] = 0.5*y*y;
	val[6] = 0.5*z*z;
	val[7] = y*z;
	val[8] = z*x;
	val[9] = x*y;

	val[10] = x*x*x/6.0;
	val[11] = y*y*y/6.0;
	val[12] = z*z*z/6.0;
	val[13] = 0.5*x*x*y;
	val[14] = 0.5*x*x*z;
	val[15] = 0.5*y*y*x;
	val[16] = 0.5*y*y*z;
	val[17] = 0.5*z*z*x;
	val[18] = 0.5*z*z*y;
	val[19] = x*y*z;

	val[20] = x*x*x*x/24.0;
	val[21] = y*y*y*y/24.0;
	val[22] = z*z*z*z/24.0;
	val[23] = x*x*x*y/6.0;
	val[24] = x*x*x*z/6.0;
	val[25] = y*y*y*x/6.0;
	val[26] = y*y*y*z/6.0;
	val[27] = z*z*z*x/6.0;
	val[28] = z*z*z*y/6.0;
	val[29] = 0.25*x*x*y*y;
	val[30] = 0.25*y*y*z*z;
	val[31] = 0.25*z*z*x*x;
	val[32] = 0.5*x*x*y*z;
	val[33] = 0.5*y*y*z*x;
	val[34] = 0.5*z*z*x*y;

	double u = 0.0;
	for(int k = 0; k < 35; k++){
		u += c[k] * val[k];
	}

	return u;
}

//=============================================================================
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#endif

