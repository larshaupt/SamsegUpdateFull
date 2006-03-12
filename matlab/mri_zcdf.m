function p = mri_zcdf(Z)
% p = mri_zcdf(Z)
% Cumulative distribution function for z, ie,
%   p = prob that a number drawn from a z distribution is <= Z.
%
% Simply returns:
%   p = (1 + erf(Z/sqrt(2)) )/2;
%
% To use this as a z-to-p converter, compute
%   p = 1 - mri_zcdf(z);
% 
% $Id: mri_zcdf.m,v 1.1 2006/03/12 18:54:19 greve Exp $

p = (1 + erf(Z/sqrt(2)) )/2;

return;




