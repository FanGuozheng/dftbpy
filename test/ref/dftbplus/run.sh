#!/bin/bash
source /opt/repo/lmod/lmod/init/profile
module load scalapack
module load gcc/6

cd $1
./dftb+ | tee output

#echo $2 >> dip.dat
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $3}' >> dip.dat
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $4}' >> dip.dat
grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $5}' >> dip.dat

#echo $2 >> bandenergy.dat
grep ' 2.00000' band.out | tail -n 1 | awk '{print $2}' >> bandenergy.dat
grep ' 0.00000' band.out | head -n 1 | awk '{print $2}' >> bandenergy.dat

grep  'Total energy:' detailed.out | awk '{print $3}' >> energy.dat

