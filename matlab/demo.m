%
% SCRIPT: DEMO
%
%   Demo usage of SG-t-SNE-Pi
%


%% CLEAN-UP

clear
close all


%% PARAMETERS

urlMNIST = 'https://github.com/daniel-e/mnist_octave/raw/master/mnist.mat';

numPCA = 50;

u         = 30;
dEmb      = 2;
lambda    = 1;
maxIter   = 1000;
earlyIter = 250;
hSide     = [];
nProc     = [];
alpha     = [];
y0        = [];


%% (BEGIN)

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

% Perform the initial dimensionality reduction using PCA
if ~isempty( numPCA )
  
  fprintf( '   - initial PCA...\n')
  
  X = bsxfun(@minus, X, mean(X, 1));
  M = pca(X,'NumComponents',numPCA,'Algorithm','svd');
  X = X * M;
  clear M;
end
    
n = size(X,1);

% initial embedding coordinates
rng(0)
y0 = 0.3*rand(dEmb, n);

fprintf( '   - DONE\n');


%% PERPLEXITY-BASED EMBEDDING (T-SNE)

fprintf( '...perplexity-based embedding (t-SNE)...\n' ); 

P = perplexityEqualize( X', u );
Y = sgtsnepi( P, dEmb, 1, maxIter, earlyIter, hSide, false, ...
              nProc, alpha, y0 );

fprintf( '   - DONE\n');

%% VISUALIZE EMBEDDING

fprintf( '...visualize embedding...\n' ); 

figure
scatter( Y(:,1), Y(:,2), eps, L, '.' )
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
% TIMESTAMP     <Jul 14, 2019: 12:31:01 Dimitris>
%
% ------------------------------------------------------------

