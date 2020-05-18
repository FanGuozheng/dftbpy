#!/bin/bash
# $1 is directory; $2 is No. of file; $3 is natom
cd $1
mpirun -np 4 ./aim | tee aims.out

echo $2 >> bandenergy.dat
grep 'Highest occupied state (VBM) at' aims.out | tail -n 1 | awk '{print $6}' >> bandenergy.dat
grep 'Lowest unoccupied state (CBM) at' aims.out | tail -n 1 | awk '{print $6}' >> bandenergy.dat

echo $2 >> vol.dat
grep 'Hirshfeld volume        :' aims.out | awk '{print $5}' >> vol.dat

echo $2 >> pol.dat
nplusone=`echo $3 + 1 | bc`
grep -A $nplusone 'C6 coefficients and polarizabilities' aims.out | tail -n $3 | awk '{print $6}' >> pol.dat

echo $2 >> dip.dat
grep 'Total dipole moment' aims.out | awk '{print $7}' >> dip.dat
grep 'Total dipole moment' aims.out | awk '{print $8}' >> dip.dat
grep 'Total dipole moment' aims.out | awk '{print $9}' >> dip.dat

