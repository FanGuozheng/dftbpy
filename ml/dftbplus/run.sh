#!/bin/bash
source /opt/repo/lmod/lmod/init/profile
module load scalapack
module load gcc/6

cd $1
./dftb+ | tee output
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $3}' >> dipole.dat
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $4}' >> dipole.dat
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $5}' >> dipole.dat

grep ' 4  ' band.out | awk '{print $2}' >> bandenergy1.dat
grep ' 5  ' band.out | awk '{print $2}' >> bandenergy2.dat

