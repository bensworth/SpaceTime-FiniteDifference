#include <iostream>
#ifndef SPACETIMEFD
	#include "SpaceTimeFD.hpp"
#endif

double IC_u(double x, double y)
{
    double temp = 0;
    if ((x + 0.25) >= 0) {
        temp += 1;
    }
    if ((x - 0.25) >= 0) {
        temp -= 1;
    }
    // std::cout << "x = " << x << ", IC_u(x) = " << temp << "\n";
    return temp;
}

double IC_v(double x, double y)
{
    return 0.0;
}


// AMG_parameters:
//  + double distance_R;
//  + std::string prerelax;
//  + std::string postrelax;
//  + int interp_type;
//  + int relax_type;
//  + int coarsen_type;
//  + double strength_tolC;
//  + double strength_tolR;
//  + double filter_tolR;
//  + double filter_tolA;
//  + int cycle_type;


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    // Do *not* use relax type 3. 
    AMG_parameters AMG = {1.5, "", "FA", 100, 10, 6, 0.1, 0.01, 0.0, 0.0, 0};
    const char* temp_prerelax = "A";
    const char* temp_postrelax = "A";

   /* Parse command line */
    int save = false;
    int use_gmres = 0;
    double hx = -1;
    double dt = -1;
    double x0 = -1;
    double x1 = 1;
    double t0 = 0;
    double t1 = 1;
    double c = 0.5;
    double cfl = -1;
    double dtdx = -1;
    int nt_loc = 200;
    int nx_loc = 20;
    int ny_loc = 0;
    int nt_glob = -1;
    int nx_glob = -1;
    int ny_glob = -1;
    int Pt = numProcess;
    int Px = 1;
    int Py = 1;
    int order = 1;

    int arg_index = 0;
    while (arg_index < argc) {
        if ( strcmp(argv[arg_index], "-nx") == 0 ) {
            arg_index++;
            nx_loc = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-ny") == 0 ) {
            arg_index++;
            ny_loc = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-nt") == 0 ) {
            arg_index++;
            nt_loc = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-gx") == 0 ) {
            arg_index++;
            nx_glob = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-gy") == 0 ) {
            arg_index++;
            ny_glob = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-gt") == 0 ) {
            arg_index++;
            nt_glob = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-Px") == 0 ) {
            arg_index++;
            Px = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-Py") == 0 ) {
            arg_index++;
            Py = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-Pt") == 0 ) {
            arg_index++;
            Pt = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-o") == 0 ) {
            arg_index++;
            order = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-dt") == 0 ) {
            arg_index++;
            dt = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-hx") == 0 ) {
            arg_index++;
            hx = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-c") == 0 ) {
            arg_index++;
            c = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-CFL") == 0 ) {
            arg_index++;
            cfl = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-dtdx") == 0 ) {
            arg_index++;
            dtdx = atof(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-gmres") == 0 ) {
            arg_index++;
            use_gmres = atoi(argv[arg_index++]);
        }
        else if ( strcmp(argv[arg_index], "-save") == 0 ) {
            arg_index++;
            save = atoi(argv[arg_index++]);
        }
        else {
            arg_index++;
        }
    }

    // If user specifies dt or hx, override default time interval
    // [0,1] or spatial problem size, respectively.
    if (dt <= 0 && nt_glob < 0) {
        nt_glob = nt_loc * Pt;
        dt = (t1 - t0) / float(nt_glob);
    }
    else if (nt_glob < 0) {
        nt_glob = nt_loc * Pt;
        t1 = t0 + int(nt_glob*dt);
    }
    if (hx <= 0 && nx_glob < 0) {
        nx_glob = nx_loc * Px;
        hx = (x1 - x0) / float(nx_glob);
    }
    else if (nx_glob < 0) {
        nx_loc = int( (x1-x0) / (Px*hx) );
        nx_glob = nx_loc * Px;
    }

    // If user specifies global t, x, or y, override local.
    if (nx_glob > 0) {
        nx_loc = int(nx_glob / Px);
        hx = (x1 - x0) / float(nx_glob);
    }
    if (nt_glob > 0) {
        nt_loc = int(nt_glob / Pt);
        if (dtdx > 0){
            dt = hx*dtdx;
            t1 = nt_glob*dt + t0;
        }
        else{
            dt = (t1 - t0) / float(nt_glob);
        }
    }

    SpaceTimeFD matrix(MPI_COMM_WORLD, nt_loc, nx_loc, Pt, Px, x0, x1, t0, t1);
	matrix.Wave1D(IC_u, IC_v, c, order);
    matrix.SetAMGParameters(AMG);
    if (use_gmres) {
        matrix.SolveGMRES(1e-8,250,3,use_gmres);
    }
    else {
        matrix.SolveAMG();
    }

    if (save) {
        matrix.SaveMatrix("A_hypre");
        matrix.SaveRHS("test_b");        
        matrix.SaveX("test_x");        
    }

    MPI_Finalize();
    return 0;
}