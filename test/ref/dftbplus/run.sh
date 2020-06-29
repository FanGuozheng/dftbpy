#!/bin/bash
cd
source .bashrc
cd -
module load scalapack
cd $1
./dftbmbd | tee output

#echo $2 >> dip.dat
#grep -A 2 'Dipole moment:' detailed.out | head -n 2 | awk '{print $3}' >> dip.dat
#grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $4}' >> dip.dat
#grep -A 2 'Dipole moment:' detailed.out | head -n 1 | awk '{print $5}' >> dip.dat
grep 'Dipole moment:' detailed.out | tail -n 1 | awk '{print $3}' >> dip.dat
grep 'Dipole moment:' detailed.out | tail -n 1 | awk '{print $4}' >> dip.dat
grep 'Dipole moment:' detailed.out | tail -n 1 | awk '{print $5}' >> dip.dat

#echo $2 >> bandenergy.dat
grep ' 2.00000' band.out | tail -n 1 | awk '{print $2}' >> bandenergy.dat
grep ' 0.00000' band.out | head -n 1 | awk '{print $2}' >> bandenergy.dat

grep  'Total energy:' detailed.out | awk '{print $3}' >> energy.dat

grep 'alpha_mbd' output | awk '{print $2}' >> poldftbplus.dat
