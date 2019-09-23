function [F,Z] = qq_exact(Y)
% TSNE_QQ - MATLAB implementation of t-SNE QQ operation
%   
% DESCRIPTION
%
%   [FREP,ZETA] = TSNE_QQ( YIN ) computes the repulsive forces FREP of a given
%   point distribution YIN. The function also return the normalization term
%   ZETA.
%  

  n = size( Y, 1 );
  
  s2 = sum(Y .^ 2, 2);

  % Student-t distribution
  R = 1 ./ (1 + s2 - 2 * (Y * Y') + s2'); 
  R(1:n+1:end) = 0;                   % set diagonal to zero

  Z = sum(R(:));
  Q = R ./ Z;   % normalize to get probabilities
  L = Q.*R;
  F = sum(L,2) .* Y - L * Y;
  
end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION       0.1
%
% TIMESTAMP     <Sep 22, 2019: 21:13:38 Dimitris>
%
% ------------------------------------------------------------
