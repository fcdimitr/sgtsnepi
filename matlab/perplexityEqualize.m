% PERPLEXITYEQUALIZE - Perform perplexity equalization (as in
% original t-SNE) and calculate the sparse stochastic matrix P
%   
% DESCRIPTION
%
%   P = PERPLEXITYEQUALIZE( X, U ) generates the sparse stochastic
%   matrix P [N-by-N] from the input data points X [L-by-N]. The
%   module computes an approximate all-kNN graph with k = 3*U using
%   FLANN and performs perplexity equalization using the desired
%   perplexity U.
%
%
  
  
