function K = computeKM(x,y,s)
% DESCRIPTION
% Compute Gaussian kernel matrix
%
%    K = computeKM(x,y,s)
%
% INPUT
%   x         samples (n1��d)
%   y         samples (n2��d)
%   s         kernel width
%
% n1,n2: number of iuput samples
% d: characteristic dimension of the samples
%
% OUTPUT
%   k         kernelMatrix (n1��n2)
%
% Created on 5th July 2019, by Kepeng Qiu.
%-------------------------------------------------------------%


sx = sum(x.^2,2);
sy = sum(y.^2,2);
K = exp((bsxfun(@minus,bsxfun(@minus,2*x*y',sx),sy'))/s^2);

end