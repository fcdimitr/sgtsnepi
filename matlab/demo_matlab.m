%
% SCRIPT: DEMO_MATLAB
%
%   Demo usage of t-SNE-Pi through MATLAB wrapper
% 
%   Before executing, issue the following commands in terminal
% 
%     cp <MROOT>/toolbox/stats/stats/private/tsnelossmex.mexmaci64 ./
%     cp <MROOT>/toolbox/stats/stats/private/tsnebhmex.mexmaci64 ./
%     cp <MROOT>toolbox/stats/stats/tsne.m ./tsne_custom.m
%     patch tsne_custom.m patch_tsnepi.patch
% 
%   where <MROOT> is the path to MATLAB installation.
%   Issue  matlabroot  to find the path
%


%% CLEAN-UP

clear
close all


%% PARAMETERS

urlMNIST = 'https://github.com/daniel-e/mnist_octave/raw/master/mnist.mat';

alg  = 'tsnepi';     % algorithm for tSNE
pca  = 50;           % PCA components (prior to tSNE)
dist = 'euclidean';  % distance metric
u    = 30;           % perplexity
dEmb = 2;            % 1, 2 and 3 dimensional embeddings are supported
verb = 2;            % verbose level


%% (BEGIN)

rng default
fprintf('\n *** begin %s ***\n\n',mfilename);

%% LOAD MNIST DIGITS

fprintf( '...load MNIST digits...\n' ); 

fprintf( '   - getting data in memory...\n')

if exist('/tmp/mnist.mat', 'file') ~= 2
  websave('/tmp/mnist.mat', urlMNIST );
end

d = load( '/tmp/mnist.mat' );
X = d.trainX;
X = im2double( X );
L = d.trainY';
clear d;
    
n = size(X,1);

fprintf( '   - DONE\n');


%% T-SNE EMBEDDING

fprintf( '...t-SNE embedding...\n' ); 

Y = tsne_custom(X, ...
                'Algorithm', alg, ...
                'NumPCAComponents', pca, ...
                'NumDimensions', dEmb, ...
                'Distance', dist, ...
                'Perplexity', u, ...
                'Verbose', verb);

fprintf( '   - DONE\n');

%% VISUALIZE EMBEDDING

fprintf( '...visualize embedding...\n' ); 

figure

switch dEmb
  case 1
    scatter(Y(:,1), Y(:,1), eps, L, '.' )
  case 2
    scatter( Y(:,1), Y(:,2), eps, L, '.' )
  case 3
    scatter3( Y(:,1), Y(:,2), Y(:,3), eps, L, '.' )
end

axis image off
colormap( jet(10) )
colorbar
title(sprintf( 't-SNE MNIST embedding | u: %d', u ) )

fprintf( '   - DONE\n');


%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);




%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION       0.1
%
% TIMESTAMP     <Sep 18, 2019: 15:45:10 Dimitris>
%
% ------------------------------------------------------------

