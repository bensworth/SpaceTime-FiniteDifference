#!/bin/bash
#MSUB -l nodes=2
#MSUB -l partition=quartz
#MSUB -l walltime=00:30:00
#MSUB -q pdebug
#MSUB -V
#MSUB -o ./results/small.out

# MPI ranks

##### These are shell commands
date
for i in `seq 1 3`
do
    
  echo "n1"
    # 20% of CFL limit
    srun -N 1 -n 1 ./driver -gmres 0 -c 0.400 -o 1 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o1_low.txt &
    srun -N 1 -n 1 ./driver -gmres 0 -c 0.247 -o 2 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o2_low.txt &
    wait
    srun -N 1 -n 1 ./driver -gmres 0 -c 0.437 -o 4 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o4_low.txt &
    
    # 70% of CFL limit
    srun -N 1 -n 1 ./driver -gmres 0 -c 1.400 -o 1 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o1_high.txt &
    wait
    srun -N 1 -n 1 ./driver -gmres 0 -c 0.865 -o 2 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o2_high.txt &
    srun -N 1 -n 1 ./driver -gmres 0 -c 1.526 -o 4 -Pt 1 -Px 1 -nt 500 -nx 500 >> ./results/n1_o4_high.txt &
    wait

  echo "n4"
    # 20% of CFL limit
    srun -N 1 -n 4 ./driver -gmres 0 -c 1.600 -o 1 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o1_low.txt &
    srun -N 1 -n 4 ./driver -gmres 0 -c 0.989 -o 2 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o2_low.txt &
    wait
    srun -N 1 -n 4 ./driver -gmres 0 -c 1.744 -o 4 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o4_low.txt &
    
    # 70% of CFL limit
    srun -N 1 -n 4 ./driver -gmres 0 -c 5.600 -o 1 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o1_high.txt &
    wait
    srun -N 1 -n 4 ./driver -gmres 0 -c 3.461 -o 2 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o2_high.txt &
    srun -N 1 -n 4 ./driver -gmres 0 -c 6.104 -o 4 -Pt 4 -Px 1 -nt 500 -nx 500 >> ./results/n4_o4_high.txt &
    wait

  echo "n16"
    # 20% of CFL limit
    srun -N 1 -n 16 ./driver -gmres 0 -c 1.600 -o 1 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o1_low.txt &
    srun -N 1 -n 16 ./driver -gmres 0 -c 0.989 -o 2 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o2_low.txt &
    wait
    srun -N 1 -n 16 ./driver -gmres 0 -c 1.744 -o 4 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o4_low.txt &
  
    # 70% of CFL limit
    srun -N 1 -n 16 ./driver -gmres 0 -c 5.600 -o 1 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o1_high.txt &
    wait
    srun -N 1 -n 16 ./driver -gmres 0 -c 3.461 -o 2 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o2_high.txt &
    srun -N 1 -n 16 ./driver -gmres 0 -c 6.104 -o 4 -Pt 8 -Px 2 -nt 500 -nx 500 >> ./results/n16_o4_high.txt &
    wait

  echo "n64"
    # 20% of CFL limit
    srun -N 2 -n 64 ./driver -gmres 0 -c 1.600 -o 1 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o1_low.txt
    srun -N 2 -n 64 ./driver -gmres 0 -c 0.989 -o 2 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o2_low.txt
    srun -N 2 -n 64 ./driver -gmres 0 -c 1.744 -o 4 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o4_low.txt

    # 70% of CFL limit
    srun -N 2 -n 64 ./driver -gmres 0 -c 5.600 -o 1 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o1_high.txt
    srun -N 2 -n 64 ./driver -gmres 0 -c 3.461 -o 2 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o2_high.txt
    srun -N 2 -n 64 ./driver -gmres 0 -c 6.104 -o 4 -Pt 16 -Px 4 -nt 500 -nx 500 >> ./results/n8_o4_high.txt

  echo RUN-${i} Done
done
