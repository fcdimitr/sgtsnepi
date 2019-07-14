% SGTSNEPI - SG-t-SNE-Pi MEX interface.
%   
% DESCRIPTION
%
%   YEMB = SGTSNEPI( P ) runs SG-t-SNE-Pi with the input graph P
%   and outputs the 2D embedding YEMB. Parameters are set to the
%   default values, see next command.
% 
%   YEMB = SGTSNEPI( P, DEMB, LAMBDA, MAXITER, EARLYITER, HSIDE, ...
%                    NPROC, ALPHA, Y0 )
%   allows the user to alter the parameters to SG-t-SNE-Pi
%
% INPUT
%
%   P           Sparse column-stochastic matrix         [n-by-n sparse]
% 
% OPTIONAL
% 
%   DEMB        Embedding dimension                     [scalar]
%               {default: 2}
%   LAMBDA      Rescaling parameter                     [scalar]
%               {default: 1}
%   MAXITER     Number of total iterations              [scalar]
%               {default: 1000}
%   EARLYITER   Number of early exaggeration iterations [scalar]
%               {default: 250}
%   HSIDE       Grid bin side length (accuracy control) [scalar]
%               {default: 0.7}
%   NPROC       Number of workers (Cilk)                [scalar]
%               (0: use default Cilk option)
%               {default: 0}
%   ALPHA       Early exaggeration coefficient          [scalar]
%               {default: 12}
%   Y0          Initial embedding coordinates           [scalar]
%               (if NULL, random points drawn from
%               a Gaussian distribution are used)
%               {default: NULL}
%
% OUTPUT
%
%   YEMB        Embedding coordinates                   [n-by-d]
%
%
  
  
  

%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION       0.1
%
% TIMESTAMP     <Jul 12, 2019: 11:23:40 Dimitris>
%
% ------------------------------------------------------------

  