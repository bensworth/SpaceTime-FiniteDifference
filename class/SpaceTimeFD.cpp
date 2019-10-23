#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
// #include "vis.c"
#include "SpaceTimeFD.hpp"


// Not sure if I have to initialize all of these to NULL. Do not do ParMatrix
// and ParVector because I think these are objects within StructMatrix/Vector. 
// 2d constructor
SpaceTimeFD::SpaceTimeFD(MPI_Comm comm, int nt, int nx, int Pt, int Px,
                         double x0, double x1, double t0, double t1) :
    m_comm{comm}, m_nt_local{nt}, m_nx_local{nx}, m_Pt{Pt}, m_Px{Px},
    m_t0{t0}, m_t1{t1}, m_x0{x0}, m_x1{x1}, m_dim(2), m_rebuildSolver{false},
    m_grid(NULL), m_graph(NULL), m_stencil_u(NULL), m_stencil_v(NULL),
    m_solver(NULL), m_gmres(NULL), m_bS(NULL), m_xS(NULL), m_AS(NULL)
{
    // Get number of processes
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProc);

    if ((m_Px*m_Pt) != m_numProc) {
        if (m_rank == 0) {
            std::cout << "Error: Invalid number of processors or processor topology \n";
            throw std::domain_error("Px*Pt != P");
        }
    }

    // Compute local indices in 2d processor array, m_px_ind and
    // m_pt_ind, from m_Px, m_Pt, and m_rank
    m_n = m_nx_local * m_nt_local;
    m_px_ind = m_rank % m_Px;
    m_pt_ind = (m_rank - m_px_ind) / m_Px;

    // TODO : should be over (m_globx + 1)?? Example had this
    m_globt = m_nt_local * m_Pt;
    m_dt = (m_t1 - m_t0) / m_globt;
    m_globx = m_nx_local * m_Px;
    m_hx = (m_x1 - m_x0) / m_globx;

    // Define each processor's piece of the grid in space-time. 
    // *IMPORTANT* - must be ordered by space and then time, so that DOFs are
    // enumerated like {(t0,x0),...,(t0,nx),(t1,x0),...}. 
    m_ilower.resize(2);
    m_iupper.resize(2);
    m_ilower[0] = m_px_ind * m_nx_local;
    m_iupper[0] = m_ilower[0] + m_nx_local-1;
    m_ilower[1] = m_pt_ind * m_nt_local;
    m_iupper[1] = m_ilower[1] + m_nt_local-1;
}


// 3d constructor
SpaceTimeFD::SpaceTimeFD(MPI_Comm comm, int nt, int nx, int ny, int Pt, int Px,
                         int Py, double x0, double x1, double y0, double y1,
                         double t0, double t1) :
    m_comm{comm}, m_nt_local{nt}, m_nx_local{nx}, m_ny_local{ny}, m_Pt{Pt},
    m_Px{Px}, m_Py{Py}, m_t0{t0}, m_t1{t1}, m_x0{x0}, m_x1{x1}, m_y0{y0},
    m_y1{y1}, m_dim(3), m_rebuildSolver{false}, m_grid(NULL), m_graph(NULL),
    m_stencil_u(NULL), m_stencil_v(NULL), m_solver(NULL), m_gmres(NULL),
    m_bS(NULL), m_xS(NULL), m_AS(NULL)
{
    // Get number of processes
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProc);

    if ((m_Px*m_Py*m_Pt) != m_numProc) {
        if (m_rank == 0) {
            std::cout << "Error: Invalid number of processors or processor topology \n";
        }
        throw std::domain_error("Px*Py*Pt != P");
    }

    // Compute local indices in 2d processor array, m_px_ind, m_py_ind, and
    // m_pt_ind, from m_Px, m_PY, m_Pt, and m_rank
    m_px_ind = m_rank % m_Px;
    m_py_ind = (( m_rank - m_px_ind) / m_Px) % m_Py;
    m_pt_ind = ( m_rank - m_px_ind - m_Px*m_py_ind) / ( m_Px*m_Py );

    // TODO : should be over (m_globx + 1)?? Example had this
    m_globt = m_nt_local * m_Pt;
    m_dt = (m_t1 - m_t0) / m_globt;
    m_globx = m_nx_local * m_Px;
    m_hx = (m_x1 - m_x0) / m_globx;
    m_globy = m_ny_local * m_Py;
    m_hy = (m_y1 - m_y0) / m_globy;

    // Define each processor's piece of the grid in space-time. 
    // *IMPORTANT* - must be ordered by space and then time, so that DOFs are
    // enumerated like {(t0,x0),...,(t0,nx),(t1,x0),...}. 
    m_ilower.resize(3);
    m_iupper.resize(3);
    m_ilower[0] = m_px_ind * m_nx_local;
    m_iupper[0] = m_ilower[0] + m_nx_local-1;
    m_ilower[1] = m_py_ind * m_ny_local;
    m_iupper[1] = m_ilower[1] + m_ny_local-1;
    m_ilower[2] = m_pt_ind * m_nt_local;
    m_iupper[2] = m_ilower[2] + m_nt_local-1;
}


SpaceTimeFD::~SpaceTimeFD()
{
    if (m_solver) HYPRE_BoomerAMGDestroy(m_solver);
    if (m_gmres) HYPRE_ParCSRGMRESDestroy(m_gmres);
    if (m_grid) HYPRE_SStructGridDestroy(m_grid);
    if (m_stencil_v) HYPRE_SStructStencilDestroy(m_stencil_v);
    if (m_stencil_u) HYPRE_SStructStencilDestroy(m_stencil_u);
    if (m_graph) HYPRE_SStructGraphDestroy(m_graph);
    if (m_AS) HYPRE_SStructMatrixDestroy(m_AS);     // This destroys parCSR matrix too
    if (m_bS) HYPRE_SStructVectorDestroy(m_bS);       // This destroys parVector too
    if (m_xS) HYPRE_SStructVectorDestroy(m_xS);
}


/* Set classical AMG parameters for BoomerAMG solve. */
void SpaceTimeFD::SetAMG()
{
   m_solverOptions.prerelax = "AA";
   m_solverOptions.postrelax = "AA";
   m_solverOptions.relax_type = 3;
   m_solverOptions.interp_type = 6;
   m_solverOptions.strength_tolC = 0.1;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = -1;
   m_solverOptions.strength_tolR = -1;
   m_solverOptions.filter_tolA = 0.0;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Set standard AIR parameters for BoomerAMG solve. */
void SpaceTimeFD::SetAIR()
{
   m_solverOptions.prerelax = "A";
   m_solverOptions.postrelax = "FFC";
   m_solverOptions.relax_type = 3;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 1.5;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filter_tolA = 0.0;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Set AIR parameters assuming triangular matrix in BoomerAMG solve. */
void SpaceTimeFD::SetAIRHyperbolic()
{
   m_solverOptions.prerelax = "A";
   m_solverOptions.postrelax = "F";
   m_solverOptions.relax_type = 10;
   m_solverOptions.interp_type = 100;
   m_solverOptions.strength_tolC = 0.005;
   m_solverOptions.coarsen_type = 6;
   m_solverOptions.distance_R = 1.5;
   m_solverOptions.strength_tolR = 0.005;
   m_solverOptions.filter_tolA = 0.0001;
   m_solverOptions.filter_tolR = 0.0;
   m_solverOptions.cycle_type = 1;
   m_rebuildSolver = true;
}


/* Provide BoomerAMG parameters struct for solve. */
void SpaceTimeFD::SetAMGParameters(AMG_parameters &params)
{
    // TODO: does this copy the structure by value?
    m_solverOptions = params;
}


void SpaceTimeFD::PrintMeshData()
{
    if (m_rank == 0) {
        std::cout << "Space-time mesh:\n\thx = " << m_hx <<
        "\n\thy = " << m_hy << "\n\tdt   = " << m_dt << "\n\n";
    }
}


/* Initialize AMG solver based on parameters in m_solverOptions struct. */
void SpaceTimeFD::SetupBoomerAMG(int printLevel, int maxiter, double tol)
{
    // If solver exists and rebuild bool is false, return
    if (m_solver && !m_rebuildSolver){
        return;
    }
    // Build/rebuild solver
    else {
        if (m_solver) {
            std::cout << "Rebuilding solver.\n";
            HYPRE_BoomerAMGDestroy(m_solver);
        }

        // Array to store relaxation scheme and pass to Hypre
        //      TODO: does hypre clean up grid_relax_points
        int ns_down = m_solverOptions.prerelax.length();
        int ns_up = m_solverOptions.postrelax.length();
        int ns_coarse = 1;
        std::string Fr("F");
        std::string Cr("C");
        std::string Ar("A");
        int* *grid_relax_points = new int* [4];
        grid_relax_points[0] = NULL;
        grid_relax_points[1] = new int[ns_down];
        grid_relax_points[2] = new int [ns_up];
        grid_relax_points[3] = new int[1];
        grid_relax_points[3][0] = 0;

        // set down relax scheme 
        for(unsigned int i = 0; i<ns_down; i++) {
            if (m_solverOptions.prerelax.compare(i,1,Fr) == 0) {
                grid_relax_points[1][i] = -1;
            }
            else if (m_solverOptions.prerelax.compare(i,1,Cr) == 0) {
                grid_relax_points[1][i] = 1;
            }
            else if (m_solverOptions.prerelax.compare(i,1,Ar) == 0) {
                grid_relax_points[1][i] = 0;
            }
        }

        // set up relax scheme 
        for(unsigned int i = 0; i<ns_up; i++) {
            if (m_solverOptions.postrelax.compare(i,1,Fr) == 0) {
                grid_relax_points[2][i] = -1;
            }
            else if (m_solverOptions.postrelax.compare(i,1,Cr) == 0) {
                grid_relax_points[2][i] = 1;
            }
            else if (m_solverOptions.postrelax.compare(i,1,Ar) == 0) {
                grid_relax_points[2][i] = 0;
            }
        }

        // Create preconditioner
        HYPRE_BoomerAMGCreate(&m_solver);
        HYPRE_BoomerAMGSetTol(m_solver, tol);    
        HYPRE_BoomerAMGSetMaxIter(m_solver, maxiter);
        HYPRE_BoomerAMGSetPrintLevel(m_solver, printLevel);
        HYPRE_BoomerAMGSetSabs(m_solver, 1);

        if (m_solverOptions.distance_R > 0) {
            HYPRE_BoomerAMGSetRestriction(m_solver, m_solverOptions.distance_R);
            HYPRE_BoomerAMGSetStrongThresholdR(m_solver, m_solverOptions.strength_tolR);
            //HYPRE_BoomerAMGSetFilterThresholdR(m_solver, m_solverOptions.filter_tolR);
        }
        HYPRE_BoomerAMGSetInterpType(m_solver, m_solverOptions.interp_type);
        HYPRE_BoomerAMGSetCoarsenType(m_solver, m_solverOptions.coarsen_type);
        HYPRE_BoomerAMGSetAggNumLevels(m_solver, 0);
        HYPRE_BoomerAMGSetStrongThreshold(m_solver, m_solverOptions.strength_tolC);
        HYPRE_BoomerAMGSetGridRelaxPoints(m_solver, grid_relax_points);
        if (m_solverOptions.relax_type > -1) {
            HYPRE_BoomerAMGSetRelaxType(m_solver, m_solverOptions.relax_type);
        }
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_coarse, 3);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_down,   1);
        HYPRE_BoomerAMGSetCycleNumSweeps(m_solver, ns_up,     2);
        if (m_solverOptions.filter_tolA > 0) {
            HYPRE_BoomerAMGSetADropTol(m_solver, m_solverOptions.filter_tolA);
        }
        // type = -1: drop based on row inf-norm
        else if (m_solverOptions.filter_tolA == -1) {
            HYPRE_BoomerAMGSetADropType(m_solver, -1);
        }

        // Do not rebuild solver unless parameters are changed.
        m_rebuildSolver = false;

        // Set cycle type for solve 
        HYPRE_BoomerAMGSetCycleType(m_solver, m_solverOptions.cycle_type);

        // Set block coarsening/interpolation
        HYPRE_BoomerAMGSetNumFunctions(m_solver, 2);
        HYPRE_BoomerAMGSetNodal(m_solver, 1);
    }
}


void SpaceTimeFD::SolveAMG(double tol, int maxiter, int printLevel)
{
    SetupBoomerAMG(printLevel, maxiter, tol);
    HYPRE_BoomerAMGSetup(m_solver, m_A, m_b, m_x);
    HYPRE_BoomerAMGSolve(m_solver, m_A, m_b, m_x);

    // Gather the solution vector
    HYPRE_SStructVectorGather(m_xS);
}


void SpaceTimeFD::SolveGMRES(double tol, int maxiter, int printLevel,
                             int precondition, int AMGiters) 
{
    HYPRE_ParCSRGMRESCreate(m_comm, &m_gmres);

    // AMG preconditioning (setup boomerAMG with 1 max iter and print level 1)
    if (precondition == 1) {
        SetupBoomerAMG(1, AMGiters, 0.0);
        HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, m_solver);
    }
    // Block-diagonal in processor preconditioning (forward solve on each proc)
    else if (precondition == 2) {
        HYPRE_GMRESSetPrecond(m_gmres, (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSolve,
                                    (HYPRE_PtrToSolverFcn) HYPRE_ParCSROnProcTriSetup, m_solver);    
    }

    HYPRE_GMRESSetKDim(m_gmres, 5);
    HYPRE_GMRESSetMaxIter(m_gmres, maxiter);
    HYPRE_GMRESSetTol(m_gmres, tol);
    HYPRE_GMRESSetPrintLevel(m_gmres, printLevel);
    HYPRE_GMRESSetLogging(m_gmres, 1);

    HYPRE_ParCSRGMRESSetup(m_gmres, m_A, m_b, m_x);
    HYPRE_ParCSRGMRESSolve(m_gmres, m_A, m_b, m_x);

    // Gather the solution vector
    HYPRE_SStructVectorGather(m_xS);
}


// TODO : add stability as puper bound on c*m_dt/m_hx
void SpaceTimeFD::GetStencil_UW1_1D(Stencil &St, double c)
{
    double k = c * m_dt / m_hx;
    if (k > 1) {
        if (m_rank == 0) {
            std::cout << "Unstable parameters: c*dt/hx = " << k << " > 1.\n";
           throw std::domain_error("Unstable integration scheme.\n");
       } 
    }
    St.uu_indices = {0, 1, 2, 3};
    St.uv_indices = {4, 5, 6};
    St.vv_indices = {0, 1, 2, 3};
    St.vu_indices = {4, 5, 6};
    St.u_data = {1.0, 
                    -k*k/2.0, // u_i-1 connection
                    -1.0+k*k,
                    -k*k/2.0, // u_i+1 connection
                    -k*m_dt/4.0, // v_i-1 connection
                    -m_dt*(2.0-k)/2.0, 
                    -k*m_dt/4.0 }; // v_i+1 connection
    St.v_data = {1.0, 
                    -k/2.0, // v_i-1 connection
                    -1.0+k, 
                    -k/2.0, // v_i+1 connection
                    -k*k/m_dt, // u_i-1 connection
                     2.0*k*k/m_dt,
                    -k*k/m_dt}; // u_i+1 connection
    St.offsets_u = {{0,0}, {-1,-1}, {0,-1}, {1,-1}, 
                            {-1,-1}, {0,-1}, {1,-1}};
    St.offsets_v = {{0,0}, {-1,-1}, {0,-1}, 
                            {1,-1}, {-1,-1}, {0,-1}, {1,-1}};
}


void SpaceTimeFD::GetStencil_UW1a_1D(Stencil &St, double c)
{
    double k = c * m_dt / m_hx;
    if (k > 0.809) {
        if (m_rank == 0) {
            std::cout << "Unstable parameters: c*dt/hx = " << k << " > 0.809.\n";
           throw std::domain_error("Unstable integration scheme.\n");
       } 
    }
    St.uu_indices = {0, 1, 2, 3};
    St.uv_indices = {4, 5, 6};
    St.vv_indices = {0, 1, 2, 3};
    St.vu_indices = {4, 5, 6};
    St.u_data = {1.0, 
                    -(4*k*k+1.0)/8.0, // u_i-1 connection
                    -(3.0-4.0*k*k)/4.0, 
                    -(4*k*k+1.0)/8.0, // u_i+1 connection 
                    -k*m_dt/4.0, // v_i-1 connection
                    -m_dt*(2.0-k)/2.0, 
                    -k*m_dt/4.0 }; // v_i+1 connection
    St.v_data = {1.0, 
                    -k/2.0, // v_i-1 connection
                    -1.0+k,
                    -k/2.0, // v_i+1 connection
                    -k*k/m_dt, // u_i-1 connection
                     2.0*k*k/m_dt, 
                    -k*k/m_dt}; // u_i-1 connection
    St.offsets_u = {{0,0}, {-1,-1}, {0,-1}, {1,-1}, 
                        {-1,-1}, {0,-1}, {1,-1}};
    St.offsets_v = {{0,0}, {-1,-1}, {0,-1}, {1,-1}, 
                        {-1,-1}, {0,-1}, {1,-1}};
}


void SpaceTimeFD::GetStencil_UW2_1D(Stencil &St, double c)
{
    double k = c * m_dt / m_hx;
    if (k > 0.618) {
        if (m_rank == 0) {
            std::cout << "Unstable parameters: c*dt/hx = " << k << " > 0.618.\n";
           throw std::domain_error("Unstable integration scheme.\n");
       } 
    }
    St.uu_indices = {0, 1, 2, 3, 4, 5};
    St.uv_indices = {6, 7, 8, 9, 10};
    St.vv_indices = {0, 1, 2, 3, 4, 5};
    St.vu_indices = {6, 7, 8, 9, 10};
        
    std::vector<double> gu = { k*k*k/4.0/m_dt,
                                 k*k/m_dt*(1.0 - k),
                                -k*k/2.0/m_dt*(4.0 - 3.0*k),
                                 k*k/m_dt*(1.0 - k),
                                 k*k*k/4.0/m_dt};
    std::vector<double> gv = {-k/8.0,
                                 k/2.0*(1.0 + k),
                                -k/4.0*(3.0 + 4.0*k),
                                 k/2.0*(1.0 + k),
                                -k/8.0};
    St.u_data = {1.0,
                    -m_dt/2.0*gu[0], // u_i-2 connection
                    -m_dt/2.0*gu[1],
                    -m_dt/2.0*gu[2]-1.0,
                    -m_dt/2.0*gu[3],
                    -m_dt/2.0*gu[4], // u_i+2 connection
                    -m_dt/2.0*gv[0], // v_i-2 connection
                    -m_dt/2.0*gv[1],
                    -m_dt/2.0*gv[2]-m_dt,
                    -m_dt/2.0*gv[3],
                    -m_dt/2.0*gv[4]}; // v_i+2 connection    
    St.v_data = {1.0,
                    -gv[0], // v_i-2 connection
                    -gv[1],
                    -gv[2]-1.0,
                    -gv[3],
                    -gv[4], // v_i+2 connection
                    -gu[0], // u_i-2 connection
                    -gu[1],
                    -gu[2],
                    -gu[3],
                    -gu[4]}; // u_i+2 connection                                
    St.offsets_u = {{0,0}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1},
                            {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}};
    St.offsets_v = {{0,0}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1},
                            {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}};        
}


void SpaceTimeFD::GetStencil_UW4_1D(Stencil &St, double c)
{
    double k = c * m_dt / m_hx;
    if (k > 1.09) {
        if (m_rank == 0) {
            std::cout << "Unstable parameters: c*dt/hx = " << k << " > 1.09.\n";
           throw std::domain_error("Unstable integration scheme.\n");
       } 
    }
    St.uu_indices = {0, 1, 2, 3, 4, 5, 6, 7};
    St.uv_indices = {8, 9, 10, 11, 12, 13, 14};
    St.vv_indices = {0, 1, 2, 3, 4, 5, 6, 7};
    St.vu_indices = {8, 9, 10, 11, 12, 13, 14};    
    St.u_data = {1.0,                
                     k*k*k/432.0*( 9.0           - 2.0*k*k), // u_i-3 connection 
                     k*k/72.0*(    3.0  - 9.0*k  - 3.0*k*k  + 2.0*k*k*k), 
                    -k*k/144.0*( 96.0 - 45.0*k - 24.0*k*k + 10.0*k*k*k), 
                    -1.0/108.0*(108.0         - 135.0*k*k + 45.0*k*k*k + 27.0*k*k*k*k - 10.0*k*k*k*k*k), 
                    -k*k/144.0*( 96.0 - 45.0*k - 24.0*k*k + 10.0*k*k*k),
                     k*k/72.0*(    3.0  - 9.0*k  - 3.0*k*k  + 2.0*k*k*k),
                     k*k*k/432.0*( 9.0           - 2.0*k*k), // u_i+3 connection
                    -m_dt*k/576.0*(   5.0            -  4.0*k*k), // v_i-3 connection
                     m_dt*k/864.0*(  45.0  + 12.0*k  - 36.0*k*k  - 8.0*k*k*k), 
                    -m_dt*k/1728.0*(225.0 + 384.0*k - 180.0*k*k - 64.0*k*k*k), 
                    -m_dt/144.0*(   144.0  - 25.0*k  - 60.0*k*k + 20.0*k*k*k + 8.0*k*k*k*k), 
                    -m_dt*k/1728.0*(225.0 + 384.0*k - 180.0*k*k - 64.0*k*k*k),
                     m_dt*k/864.0*(  45.0  + 12.0*k  - 36.0*k*k  - 8.0*k*k*k), 
                    -m_dt*k/576.0*(   5.0             - 4.0*k*k)}; // v_i+3 connection
    St.v_data = {1.0,        
                    -k/288.0*(  5.0           - 8.0*k*k), // v_i-3 connection
                     k/48.0*(   5.0  + 2.0*k  - 8.0*k*k  - 2.0*k*k*k), 
                    -k/96.0*(  25.0 + 64.0*k - 40.0*k*k - 16.0*k*k*k), 
                    -1.0/72.0*(72.0 - 25.0*k - 90.0*k*k + 40.0*k*k*k + 18.0*k*k*k*k), 
                    -k/96.0*(  25.0 + 64.0*k - 40.0*k*k - 16.0*k*k*k), 
                     k/48.0*(   5.0  + 2.0*k  - 8.0*k*k  - 2.0*k*k*k), 
                    -k/288.0*(  5.0           - 8.0*k*k), // v_i+3 connection
                     k*k*k/48.0/m_dt*(3.0               - k*k), // u_i-3 connection
                     k*k/24.0/m_dt*(  2.0  - 9.0*k  - 4.0*k*k +  3.0*k*k*k), 
                    -k*k/48.0/m_dt*( 64.0 - 45.0*k - 32.0*k*k + 15.0*k*k*k), 
                     k*k/12.0/m_dt*( 30.0 - 15.0*k - 12.0*k*k +  5.0*k*k*k), 
                    -k*k/48.0/m_dt*( 64.0 - 45.0*k - 32.0*k*k + 15.0*k*k*k), 
                     k*k/24.0/m_dt*(  2.0  - 9.0*k  - 4.0*k*k +  3.0*k*k*k), 
                     k*k*k/48.0/m_dt*(3.0               - k*k)}; // u_i+3 connection
    St.offsets_u = {{0,0}, {-3,-1}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}, {3,-1},
                            {-3,-1}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}, {3,-1}};
    St.offsets_v = {{0,0}, {-3,-1}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}, {3,-1},
                            {-3,-1}, {-2,-1}, {-1,-1}, {0,-1}, {1,-1}, {2,-1}, {3,-1}};
}


void SpaceTimeFD::GetStencil_UW1_2D(Stencil &St, double c)
{
    // TODO
}


void SpaceTimeFD::GetStencil_UW2_2D(Stencil &St, double c)
{
    // TODO
}


// First-order upwind discretization of the 1d-space, 1d-time homogeneous
// wave equation. Initial conditions (t = 0) are passed in function pointers
// IC_u(double x) and IC_v(double x), which give the ICs for u and v at point
// (0,x). 
//
// TODO : Can we make polymorphism for function tha takes IC_u and IC_v as
// functions of one parameter (for 1D space) and passes them to this function
// with dummy arguments or something as second variable?
void SpaceTimeFD::Wave1D(double (*IC_u)(double, double),
                         double (*IC_v)(double, double), 
                         double c,
                         int order,
                         bool alternate)
{
    Stencil St;
    if (m_dim == 2) {
        if (order == 1) {
            if (alternate) {
                GetStencil_UW1a_1D(St, c);
            }
            else {
                GetStencil_UW1_1D(St, c);
            }         
        }
        else if (order == 2) {
            GetStencil_UW2_1D(St, c);
        }
        else if (order == 4) {
            GetStencil_UW4_1D(St, c);            
        }
        else {
            if (m_rank == 0) {
                throw std::domain_error("Only orders 1, 1a, 2, and 4 available in 1D.\n");
            }
        }
        if (m_rank == 0) {
            std::cout << "    1d Space-time wave equation, order-" << order << ":\n" <<
               "        c*dt/dx  = (" << c * m_dt / m_hx << "\n" << 
               "        (dx, dt) = (" << m_hx << ", " << m_dt << ")\n" << 
               "        (nx, nt) = (" << m_globx << ", " << m_globt << ")\n" << 
               "        (Px, Pt) = (" << m_Px << ", " << m_Pt << ")\n";
        }
    }
    else if (m_dim == 3) {
        if (order == 1) {
            GetStencil_UW1_2D(St, c);
        }
        else if (order == 2) {
            GetStencil_UW2_2D(St, c);
        }
        else {
            if (m_rank == 0) {
                throw std::domain_error("Only orders 1, and 2 available in 2D.\n");
            }
        }
        if (m_rank == 0) {
            std::cout << "    2d Space-time wave equation, order-" << order << ":\n" <<
               "        (dx, dy, dt) = " << m_hx << ", " << m_hy << ", " << m_dt << ")\n" << 
               "        (nx, ny, nt) = (" << m_globx << ", " << m_globy << ", " << m_globt << ")\n" << 
               "        (Px, Py, Pt) = (" << m_Px << ", " << m_Py << ", " << m_Pt << ")\n";
        }       
    }

    // Create an empty 2D grid object with 1 part
    HYPRE_SStructGridCreate(m_comm, m_dim, 1, &m_grid);

    // Add this processor's box to the grid (on part 0)
    HYPRE_SStructGridSetExtents(m_grid, 0, &m_ilower[0], &m_iupper[0]);

    // Define two variables on grid (on part 0)
    HYPRE_SStructVariable vartypes[2] = {HYPRE_SSTRUCT_VARIABLE_CELL,
                            HYPRE_SSTRUCT_VARIABLE_CELL };
    HYPRE_SStructGridSetVariables(m_grid, 0, 2, vartypes);

    /* ------------------------------------------------------------------
    *                  Add periodic boundary conditions
    * ---------------------------------------------------------------- */
    // Set periodic on *all* processors
    std::vector<int> periodic(m_dim,0);
    periodic[0] = m_globx; // periodic in x
    if (m_dim == 3) {
        periodic[1] = m_globy; // periodic in y for 2d-space
    }
    HYPRE_SStructGridSetPeriodic(m_grid, 0, &periodic[0]); 

    // Finalize grid assembly.
    HYPRE_SStructGridAssemble(m_grid);

    /* ------------------------------------------------------------------
    *                   Define discretization stencils
    * ---------------------------------------------------------------- */
    int n_uu_stenc      = (St.uu_indices).size();
    int n_uv_stenc      = (St.uv_indices).size();
    int stencil_size_u  = n_uu_stenc + n_uv_stenc;
    HYPRE_SStructStencilCreate(m_dim, stencil_size_u, &m_stencil_u);

    // Set stencil for u-u connections (variable 0)
    for (auto entry : (St.uu_indices)) {
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &(St.offsets_u)[entry][0], 0);
    }

    // Set stencil for u-v connections (variable 1)
    for (auto entry : (St.uv_indices)) {
        HYPRE_SStructStencilSetEntry(m_stencil_u, entry, &(St.offsets_u)[entry][0], 1);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_uu = n_uu_stenc * m_n;
    std::vector<double> uu_values(n_uu);
    for (int i=0; i<n_uu; i+=n_uu_stenc) {
        for (int j=0; j<n_uu_stenc; j++) {
            uu_values[i+j] = (St.u_data)[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_uv = n_uv_stenc * m_n;
    std::vector<double> uv_values(n_uv);
    for (int i=0; i<n_uv; i+=n_uv_stenc) {
        for (int j=0; j<n_uv_stenc; j++) {
            uv_values[i+j] = (St.u_data)[j+n_uu_stenc];
        }
    }

    // Stencil object for variable v (labeled as variable 1).
    int n_vv_stenc     = (St.vv_indices).size();
    int n_vu_stenc     = (St.vu_indices).size();
    int stencil_size_v = n_vv_stenc + n_vu_stenc;
    HYPRE_SStructStencilCreate(m_dim, stencil_size_v, &m_stencil_v);

    // Set stencil for v-v connections (variable 1)
    for (auto entry : (St.vv_indices)) {
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &(St.offsets_v)[entry][0], 1);
    }

    // Set stencil for v-u connections (variable 0)
    for (auto entry : (St.vu_indices)) {
        HYPRE_SStructStencilSetEntry(m_stencil_v, entry, &(St.offsets_v)[entry][0], 0);
    }

    // Set u-stencil entries (to be added to matrix later). Note that
    // HYPRE_SStructMatrixSetBoxValues can only set values corresponding
    // to stencil entries for one variable at a time
    int n_vv = n_vv_stenc * m_n;
    std::vector<double> vv_values(n_vv);
    for (int i=0; i<n_vv; i+=n_vv_stenc) {
        for (int j=0; j<n_vv_stenc; j++) {
            vv_values[i+j] = (St.v_data)[j];
        }
    }

    // Fill in stencil for u-v entries here (to be added to matrix later)
    int n_vu = n_vu_stenc * m_n;
    std::vector<double> vu_values(n_vu);
    for (int i=0; i<n_vu; i+=n_vu_stenc) {
        for (int j=0; j<n_vu_stenc; j++) {
            vu_values[i+j] = (St.v_data)[n_vv_stenc+j];
        }
    }

    /* ------------------------------------------------------------------
    *                      Fill in sparse matrix
    * ---------------------------------------------------------------- */
    HYPRE_SStructGraphCreate(m_comm, m_grid, &m_graph);
    HYPRE_SStructGraphSetObjectType(m_graph, HYPRE_PARCSR);
    HYPRE_SStructGraphSetObjectType(m_graph, HYPRE_PARCSR);

    // Assign the u-stencil to variable u (variable 0), and the v-stencil
    // variable v (variable 1), both on part 0 of the m_grid
    HYPRE_SStructGraphSetStencil(m_graph, 0, 0, m_stencil_u);
    HYPRE_SStructGraphSetStencil(m_graph, 0, 1, m_stencil_v);

    // Assemble the m_graph
    HYPRE_SStructGraphAssemble(m_graph);

    // Create an empty matrix object
    HYPRE_SStructMatrixCreate(m_comm, m_graph, &m_AS);
    HYPRE_SStructMatrixSetObjectType(m_AS, HYPRE_PARCSR);
    HYPRE_SStructMatrixInitialize(m_AS);

    // Set values in matrix for part 0 and variables 0 (u) and 1 (v)
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0], 0,
                                    n_uu_stenc, &(St.uu_indices)[0], &uu_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0], 0,
                                    n_uv_stenc, &(St.uv_indices)[0], &uv_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0], 1,
                                    n_vv_stenc, &(St.vv_indices)[0], &vv_values[0]);
    HYPRE_SStructMatrixSetBoxValues(m_AS, 0, &m_ilower[0], &m_iupper[0], 1,
                                    n_vu_stenc, &(St.vu_indices)[0], &vu_values[0]);

    /* ------------------------------------------------------------------
    *                      Construct linear system
    * ---------------------------------------------------------------- */
    // Finalize matrix assembly
    HYPRE_SStructMatrixAssemble(m_AS);

    // Create an empty vector object
    HYPRE_SStructVectorCreate(m_comm, m_grid, &m_bS);
    HYPRE_SStructVectorCreate(m_comm, m_grid, &m_xS);

    // Set vectors to be par csr type
    HYPRE_SStructVectorSetObjectType(m_bS, HYPRE_PARCSR);
    HYPRE_SStructVectorSetObjectType(m_xS, HYPRE_PARCSR);

    // Indicate that vector coefficients are ready to be set
    HYPRE_SStructVectorInitialize(m_bS);
    HYPRE_SStructVectorInitialize(m_xS);

    // Set right hand side and inital guess. RHS is nonzero only at time t=0
    // because we are solving homogeneous wave equation. Because scheme is
    // explicit, set solution equal to rhs there because first t rows are
    // diagonal. Otherwise, rhs = 0 and we use 0 initial guess.
    std::vector<double> rhs(m_n, 0);    
    if (m_dim == 2) {
        if (m_pt_ind == 0) {
            for (int i=0; i<m_nx_local; i++) {
                double temp_x = m_x0 + (m_px_ind*m_nx_local + i) * m_hx;
                rhs[i] = IC_u(temp_x,0);
            }
        }
        HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);
        HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);

        if (m_pt_ind == 0) {
            for (int i=0; i<m_nx_local; i++) {
                double temp_x = m_x0 + (m_px_ind*m_nx_local + i) * m_hx;
                rhs[i] = IC_v(temp_x,0);
            }
        }
        HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
        HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
    }
    // TODO : verify that this is correct
    else if (m_dim == 3) {
        if (m_pt_ind == 0) {
            for (int j=0; j<m_ny_local; j++) {
                double temp_y = m_y0 + (m_py_ind*m_ny_local + j) * m_hy;
                for (int i=0; i<m_nx_local; i++) {
                    double temp_x = m_x0 + (m_px_ind*m_nx_local + i) * m_hx;
                    rhs[j*m_ny_local + i] = IC_u(temp_x, temp_y);
                }
            }
        }
        HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);
        HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 0, &rhs[0]);

        if (m_pt_ind == 0) {
            for (int j=0; j<m_ny_local; j++) {
                double temp_y = m_y0 + (m_py_ind*m_ny_local + j) * m_hy;
                for (int i=0; i<m_nx_local; i++) {
                    double temp_x = m_x0 + (m_px_ind*m_nx_local + i) * m_hx;
                    rhs[j*m_ny_local + i] = IC_v(temp_x, temp_y);
                }
            }
        }
        HYPRE_SStructVectorSetBoxValues(m_bS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
        HYPRE_SStructVectorSetBoxValues(m_xS, 0, &m_ilower[0], &m_iupper[0], 1, &rhs[0]);
    }

    // Finalize vector assembly
    HYPRE_SStructVectorAssemble(m_bS);
    HYPRE_SStructVectorAssemble(m_xS);

    // Get objects for sparse matrix and vectors.
    HYPRE_SStructMatrixGetObject(m_AS, (void **) &m_A);
    HYPRE_SStructVectorGetObject(m_bS, (void **) &m_b);
    HYPRE_SStructVectorGetObject(m_xS, (void **) &m_x);
}




#if 0
// Save the solution for GLVis visualization, see vis/glvis-ex7.sh
// TODO : Fix, add numvariables as option? 
SpaceTimeFD::VisualizeSol()
{
    FILE *file;
    char filename[255];

    int k, part = 0, var;
    int m_n = n*n;
    double *values = (double*) calloc(m_n, sizeof(double));

    /* save local solution for variable u */
    var = 0;
    HYPRE_SStructVectorGetBoxValues(m_xS, 0, m_ilower, m_iupper,
                                    var, values);

    sprintf(filename, "%s.%06d", "vis/ex9-u.sol", m_rank);
    if ((file = fopen(filename, "w")) == NULL) {
        printf("Error: can't open output file %s\n", filename);
        MPI_Finalize();
        exit(1);
    }

    /* save solution with global unknown numbers */
    int k = 0;
    for (int j=0; j<n; j++) {
        for (int i=0; i<n; i++){
            fprintf(file, "%06d %.14e\n", pj*N*n*n+pi*n+j*N*n+i, values[k++]);
        }
    }

    fflush(file);
    fclose(file);

    /* save local solution for variable v */
    var = 1;
    HYPRE_SStructVectorGetBoxValues(m_xS, 0, m_ilower, m_iupper,
                                    var, values);

    sprintf(filename, "%s.%06d", "vis/ex9-v.sol", m_rank);
    if ((file = fopen(filename, "w")) == NULL) {
        printf("Error: can't open output file %s\n", filename);
        MPI_Finalize();
        exit(1);
    }

    /* save solution with global unknown numbers */
    k = 0;
    for (int j=0; j<n; j++) {
        for (int i=0; i<n; i++) {
            fprintf(file, "%06d %.14e\n", pj*N*n*n+pi*n+j*N*n+i, values[k++]);
        }
    }

    fflush(file);
    fclose(file);

    free(values);

    /* save global finite element mesh */
    if (m_rank == 0){
        GLVis_PrintGlobalSquareMesh("vis/ex9.mesh", N*n-1);
    }
}
#endif