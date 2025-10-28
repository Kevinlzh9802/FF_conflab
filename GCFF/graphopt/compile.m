% mex  -v CXXFLAGS="-O3 -ffast-math  \$CXXFLAGS" allgc.cpp  
% mex  -v CXXFLAGS="-O3 -ffast-math  \$CXXFLAGS" expand.cpp 
% mex  -v CXXFLAGS="-O3 -ffast-math  \$CXXFLAGS" multi.cpp 
%mex   -v CXXFLAGS="-O3 -ffast-math  \$CXXFLAGS" convex.cpp 
% mex allgc.cpp 
% mex expand.cpp -DDEBUG_EXPAND
% mex multi.cpp 

% mex -v CXXFLAGS="$CXXFLAGS -DDEBUG_EXPAND -DDEBUG_SOLVE -O2" expand.cpp
mex -v CXXFLAGS="$CXXFLAGS -O2" expand.cpp
%load("/homes/jrkf/Public/pre_gc.mat", "K","s","old_labels","Lambda","Np","MDL")