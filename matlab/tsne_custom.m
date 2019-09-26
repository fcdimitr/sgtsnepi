function [Y,loss,time] = tsne_custom(X,varargin)
%TSNE t-Distributed Stochastic Neighbor Embedding.
%   Y = tsne(X) returns the representation of the N by P matrix X in the
%   two dimensional space. Each row in X represents an observation. Rows
%   with NaN missing values are removed.
%
%   [Y,loss] = tsne(X) returns the loss of using the joint distribution of 
%   Y to represent the joint distribution of X. The loss is measured by the
%   Kullback-Leibler divergence between the joint distributions of X and Y.
%
%   [...] = TSNE(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies optional
%   parameter name/value pairs to control further details of TSNE.  
%   Parameters are: 
%
%   'Algorithm'    - Algorithm that TSNE uses to find Y. Choices are:
%      'barneshut' - Approximate computation for the joint distributions of 
%                    X and the gradient of the optimization. TSNE first 
%                    search for 3*Perplexity(see below) nearest neighbors 
%                    and use them to approximate the joint distributions of
%                    X. These nearest neighbors and the Barnes-Hut algorithm
%                    are used to approximate the gradient (default).
%      'exact'     - Exact computation for the joint distributions of
%                    X and the gradient of the optimization.
%
%   'Distance'     - A string specifies the metric of distance when 
%                    calculating distance between observations.
%                    Default: 'euclidean'.
%       'euclidean'    - Euclidean distance.
%       'seuclidean'   - Standardized Euclidean distance. Each
%                        coordinate difference between rows in X is
%                        scaled by dividing by the corresponding element
%                        of the standard deviation computed from X,
%                        S=nanstd(X).
%       'cityblock'    - City block metric.
%       'minkowski'    - Minkowski distance, with exponent 2. 
%       'chebychev'    - Chebychev distance (maximum coordinate difference).
%       'mahalanobis'  - Mahalanobis distance, using the sample 
%                        covariance of X as computed by nancov. 
%       'cosine'       - One minus the cosine of the included angle 
%                        between points (treated as vectors).
%       'correlation'  - One minus the sample correlation between points
%                        (treated as sequences of values).
%       'spearman'     - One minus the sample Spearman's rank correlation
%                        between observations, treated as sequences of values.
%       'hamming'      - Hamming distance, the percentage of coordinates 
%                        that differ.
%       'jaccard'      - One minus the Jaccard coefficient, the percentage
%                        of nonzero coordinates that differ.
%       function       - A distance function specified using @:
%            A distance function must be of the form
%            function D2 = distfun(ZI, ZJ)
%            taking as arguments a 1-by-n vector ZI containing a single
%            observation from X or Y, an m2-by-n matrix ZJ containing
%            multiple observations from X or Y, and returning an m2-by-1
%            vector of distances D2, whose Jth element is the distance
%            between the observations ZI and ZJ(J,:). If your data is not
%            sparse, generally it is faster to use a built-in distance than
%            to use a function handle.
%
%      For 'barneshut' algorithm, TSNE uses knnsearch to find the nearest
%      neighbors for each observation and compute the pairwise distances 
%      from the nearest neighbors. For both algorithms, TSNE uses squared 
%      pairwise distances to calculate the Gaussian kernel in the joint 
%      distribution of X.
%
%   'NumDimensions'- A positive integer specifying the number of dimension  
%                    of the representation Y. Default: 2
%   'NumPCAComponents' - A nonnegative integer specifying the number of PCA
%                    components. If the value is positive, TSNE first apply
%                    PCA to reduce the dimensionality of X to the specified
%                    number before learning the representation Y. If the 
%                    value is 0, TSNE does not perform PCA. Default: 0
%   'InitialY'     - A N by D matrix of initial points of Y with D being the 
%                    dimension of the representation Y. 
%                    Default: 1e-4*randn(N,D)
%   'Perplexity'   - A positive scalar representing the effective number 
%                    of local neighbors of each observation. Larger perplexity 
%                    makes 'barneshut' algorithm use more points as nearest
%                    neighbors. Use a larger value of perplexity for larger 
%                    dataset. Typical values are between 5 and 50. The number
%                    of nearest neighbors (not including the point itself)
%                    used in 'barneshut' algorithm is chosen to be the smaller
%                    value of 3*Perplexity and N-1. Default: 30
%   'Exaggeration' - A positive scalar no less than 1 specifying the tightness
%                    of the natural clusters in X at the start of the optimization
%                    (before iteration 100). A large exaggeration makes TSNE
%                    learn larger joint probabilities of Y and creates 
%                    relatively more space between clusters in Y. If the 
%                    value of KL divergence increases in the early stage of
%                    the optimization, reducing the exaggeration to a smaller
%                    number may help.  Default: 4
%   'LearnRate'    - A positive scalar specifying the learning rate of the 
%                    optimization process. Typical values are between 100
%                    and 1000. If the learning rate is too small, the
%                    optimization process may get stuck in a bad local
%                    minimum. If the value of KL divergence increases, try
%                    to set the learning rate to a smaller number.
%                    Default: 500
%   'Theta'        - A nonnegative scalar between 0 and 1 specifying the 
%                    trade-off of speed and accuracy of 'barneshut' algorithm. 
%                    Larger value of THETA leads to coarser approximation 
%                    in the gradient calculation and produce result with 
%                    less accuracy but faster learning. Theta applies only 
%                    to the 'barneshut' algorithm. Default: 0.5
%   'Standardize'  - Logical scalar. If true, standardize X by centering
%                    and dividing columns by their standard deviations. If 
%                    features in X are on different scales, 'Standardize'
%                    should be set to true because the learning process is
%                    based on nearest neighbors and features with large
%                    scales can override the contribution of features with
%                    small scales. Default: false
%   'NumPrint'     - A positive integer specifying the frequency with which 
%                    to display convergence summary on screen. Default: 20
%   'Options'      - A structure containing optimization options with the 
%                    following fields:
%          'MaxIter'   - Maximum number of iterations to take. Default: 1000
%          'TolFun'    - Termination tolerance for the gradient of the KL 
%                        divergence function.  Default: 1e-10
%          'OutputFcn' - Function handle specified using @, a cell array
%                        of function handles or an empty array (default).
%                        TSNE calls function(s) in 'OutputFcn' at every 
%                        'NumPrint' iterations.
%   'Verbose'      - 0, 1 or 2. Controls the level of detail of command 
%                    line display. Default: 0.
%                           0: Do not display anything
%                           1: Display the convergence summary every 
%                              'NumPrint' iterations.
%                           2: Same as (1), but also displays the current
%                              state of the learning process, and more. 
%                              The message of the variances of Gaussian 
%                              kernels (used in the computation of the joint 
%                              probability of X) can be used in the diagnosis
%                              of the learning process. A large difference 
%                              in the scales of the minimum and maximum
%                              variances may indicate X containing some 
%                              large values. Rescaling X may help.
%
%   Example:
%   load('FisherIris.mat')
%   rng('default');
%   Y = tsne(meas,'Algorithm','exact','Standardize',true,'Perplexity',20);
%   % Plot the result
%   figure;
%   gscatter(Y(:,1),Y(:,2),species);
%
%   See also pca, pdist, knnsearch, statset, gscatter, scatter

% References:
%   [1] Hinton, Geoffrey E., and Sam T. Roweis, Stochastic neighbor embedding,
%       Advances in neural information processing systems (2002).
%   [2] Van der Maaten, Laurens, and Geoffrey Hinton, Visualizing data using
%       t-SNE, Journal of Machine Learning Research 9.2579-2605 (2008): 85.
%   [3] Van Der Maaten, Laurens, Fast Optimization for t-SNE, In Neural 
%       Information Processing Systems (NIPS) 2010 Workshop on Challenges
%       in Data Visualization, Vol. 100, (2010).
%   [4] Jacobs, Robert A, Increased rates of convergence through learning 
%       rate adaptation, Neural networks 1.4 (1988): 295-307.
%   [5] https://lvdmaaten.github.io/tsne/

%   Copyright 2016-2018 The MathWorks, Inc.


if nargin > 1
    [varargin{:}] = convertStringsToChars(varargin{:});
end

if nargin<1
    error(message('stats:tsne:TooFewInputs'));
end

paramNames = {'Algorithm',  'Distance',   'NumDimensions',  'NumPCAComponents',...
              'InitialY',   'Perplexity', 'Exaggeration',   'LearnRate',...
              'Theta',      'Standardize','NumPrint',       'options',    'Verbose'};
defaults   = {'barneshut',  'euclidean',   [],               0,...     
               [],          [],            4,               500,...
               [],         false,        20,               [],            0};

tsnepiopt.buildcsb    = 0;
tsnepiopt.destroycsb  = 1;
tsnepiopt.computegrad = 2;

[algorithm, distance, ydims, numPCA, ystart, perplexity, exaggeration, learnrate, theta,...
  standardize, numprint, options, verbose] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

% Input Checking
internal.stats.checkSupportedNumeric('X',X,false,false,false);
if ~ismatrix(X)
    error(message('stats:tsne:BadX'));
end
p = size(X,2);
if any(isinf(X(:)))
    error(message('stats:tsne:InfX'));
end
    
if ~isempty(ystart)
    internal.stats.checkSupportedNumeric('InitialY',ystart,false,false,false);
    % check Inf values
    if any(isinf(ystart(:)))
        error(message('stats:tsne:InfInitialY'));
    end
    if size(X,1)~=size(ystart,1)
        error(message('stats:tsne:InputSizeMismatch'));
    end
    ystartcols = size(ystart,2);
    if ~isempty(ydims) 
        if ystartcols~=ydims
            error(message('stats:tsne:BadInitialY'));
        end
    elseif ystartcols>p
        error(message('stats:tsne:BadYdims1'));
    else
        ydims = ystartcols;
    end
else
    if isempty(ydims)
        ydims = min(p,2);
    elseif ~internal.stats.isScalarInt(ydims,1)
        error(message('stats:tsne:InvalidYdims'));
    elseif ydims>p
        error(message('stats:tsne:BadYdims2'));
    end
    ystart = 1e-4*randn(size(X,1), ydims);
end
ystart = cast(ystart,'like',X);

% Remove NaN rows, if any
haveNaN = false;
if any(any(isnan(X))) || any(any(isnan(ystart)))
   haveNaN = true;
   [~,~,X,ystart] = statremovenan(X,ystart);
   if isempty(X)
       warning(message('stats:tsne:EmptyXafterNaN'));
   else
       warning(message('stats:tsne:NaNremoved'));
   end
end

N = size(X,1);
if ~internal.stats.isScalarInt(numPCA,0)
    error(message('stats:tsne:InvalidNumPCA','NumPCAComponents'));
elseif numPCA>0 && (numPCA<ydims || numPCA>p)
    error(message('stats:tsne:BadNumPCA','NumPCAComponents'));
end

if ~(isFiniteRealNumericScalar(exaggeration) && exaggeration>=1)
    error(message('stats:tsne:BadExaggeration'));
end
if ~(isFiniteRealNumericScalar(learnrate) && learnrate>0)
    error(message('stats:tsne:BadLearnRate'));
end

if ~internal.stats.isScalarInt(numprint,1)
    error(message('stats:tsne:BadNumPrint'));
end

if ~(isFiniteRealNumericScalar(verbose) && ismember(verbose,[0 1 2]))
    error(message('stats:tsne:BadVerbose'));
end

options = statset(statset('tsne'), options);

AlgorithmNames = {'exact','barneshut','tsnepi'};
algorithm = internal.stats.getParamVal(algorithm,AlgorithmNames,...
    '''Algorithm''');

if ~isscalar(standardize) || (~islogical(standardize) && standardize~=0 && standardize~=1)
    error(message('stats:tsne:InvalidStandardize'));
end

if ~isempty(perplexity) 
    if ~(isFiniteRealNumericScalar(perplexity) && perplexity>0)
        error(message('stats:tsne:BadPerplexity'));
    elseif ~haveNaN && perplexity>N
        error(message('stats:tsne:LargePerplexity'));
    elseif haveNaN && perplexity>N
        error(message('stats:tsne:LargePerplexityAfterRemoveNaN'));
    end
else
    perplexity = min(ceil(N/2),30);
end

if ~(isempty(theta)|| (isFiniteRealNumericScalar(theta) && theta<=1 && theta>=0))
    error(message('stats:tsne:BadTheta'));
end
if strcmpi(algorithm,'exact') 
    if ~isempty(theta) 
        error(message('stats:tsne:InvalidTheta'));
    end
else
    if isempty(theta)
        theta = 0.5;
    end
end

% Handle empty case
if isempty(X)
    Y = zeros(N,ydims,'like',X);
    loss = cast([],'like',X);
    return;
end

% Standardize data
if standardize
    constantCols = (range(X,1)==0);
    sigmaX = std(X,0,1);
    % Avoid dividing by zero with constant columns
    sigmaX(constantCols) = 1;
    X = (X-mean(X,1))./sigmaX;
end

% Perform PCA
if numPCA>0
    if verbose > 1
        fprintf('%s\n',getString(message('stats:tsne:PerformPCA',num2str(numPCA))));
    end
    [~,X] = pca(X,'Centered',false,'Economy',false,'NumComponents',numPCA);
end

if strcmpi(algorithm,'exact')
    if verbose>1
        fprintf('%s\n',getString(message('stats:tsne:ComputeDistMat')));
    end
    if N==1
        % Only one observation
        tempDistMat = 0;
    else
        tempDistMat = pdist(X,distance);
        tempDistMat = squareform(tempDistMat);
        tempDistMat = tempDistMat.^2;
    end
    if verbose > 1
        fprintf('%s\n',getString(message('stats:tsne:ComputeProbMat')));
    end
    [probMatX,sig2] = binarySearchVariance(tempDistMat,perplexity);
    colidx = [];
    rowcnt = [];
    % Compute joint probability and set the diagnals to be 0
    probMatX(1:N+1:end) = 0;
    probMatX = (probMatX + probMatX')/(2*N);
else
    if verbose>1
        fprintf('%s\n',getString(message('stats:tsne:PerformKnnSearch')));
    end
    % Find nearest neighbors of each data point
    ns = createns(X,'distance',distance);
    k = min(N, 3 * floor(perplexity)+1);
    if k==0
        % Empty input
        knnidx = [];
        D = [];
    else
        [knnidx,D] = knnsearch(ns,X,'k',k);
        knnidx(:,1) = [];
    end
    if verbose > 1
        fprintf('%s\n',getString(message('stats:tsne:ComputeProbMat')));
    end
    D = D(:,2:end).^2;
    K = size(D,2);
    maxDensity = 0.4;
    % Compute probMatX using knn results
    if (2*K)/N<maxDensity
        % If density of matrix less than 0.4, only return N by K probMatX
        [probMatX,sig2] = binarySearchVariance(D,perplexity);
    else
        % Otherwise, return full matrix
        [probMatX,sig2] = binarySearchVariance(D,perplexity,knnidx);
    end
    % Find the nonzero elements and their indices in probMatX
    [colidx, rowcnt, probMatX] = probMatXknn(probMatX,knnidx);
end
clear X tempDistMat D

if any(probMatX(:)<0 | probMatX(:)>1)
    error(message('stats:tsne:BadJointProb'));
end
probMatX = max(probMatX,realmin(class(ystart)));

if any(sig2<0)
    error(message('stats:tsne:BadVariance'));
end
% Display diagnosis message to command window
if verbose>1
    sig2 = 1./sig2;
    if isempty(sig2)
        avgSig2=[];
    else
        avgSig2 = mean(sig2);
    end
    minSig2 = min(sig2);
    maxSig2 = max(sig2);
    fprintf('%s\n',getString(message('stats:tsne:MeanVariance',num2str(avgSig2))));
    fprintf('%s\n',getString(message('stats:tsne:MinVariance',num2str(minSig2))));
    fprintf('%s\n',getString(message('stats:tsne:MaxVariance',num2str(maxSig2))));
end

% Perform t-SNE to find Y
if verbose>1
    fprintf('%s\n',getString(message('stats:tsne:PerformTSNE')));
end

[Y,loss,time] = tsneEmbedding(ystart,probMatX,exaggeration,learnrate,...
    numprint,verbose,options,algorithm,theta,colidx,rowcnt,tsnepiopt);
end % tsne

% ---------------------------------------------------
% SUBFUNCTIONS 
% ---------------------------------------------------
function t = isFiniteRealNumericScalar(x)
%   T = ISSCALARINT(X) returns true if X is a finite numeric real
%   scalar value, and false otherwise.
t = isscalar(x) && isnumeric(x) && isreal(x) && isfinite(x);
end

function [condProbMatX,sig2] = binarySearchVariance(D,perplexity,varargin)
% Binary search for the sigma of the conditional probability
[N,K] = size(D);
if nargin > 2
    knnidx = varargin{:};
    condProbMatX = zeros(N);
else
     condProbMatX = zeros(N,K);
end
sig2 = ones(N,1);
H = log(perplexity);

tolBinary = 1e-5;
maxit = 100;
notConverge = false(N,1);

for i = 1:N
    a = -Inf;
    c = Inf;
    iter = 0;
    while(true)
        P_i = exp(-D(i,:)*sig2(i));
        if K==N
            P_i(i) = 0;
        end
        sum_i = max(sum(P_i),realmin(class(D)));
        P_i = P_i./sum_i;
        H_i = log(sum_i) + sig2(i)*sum(D(i,:).*P_i);
        fval = H_i - H;
        if abs(fval)< tolBinary
            break;
        end
        if fval > 0
            a = sig2(i);
            if isinf(c)
                sig2(i) = 2*sig2(i);
            else
                sig2(i) = 0.5*(sig2(i) + c);
            end
        else
            c = sig2(i);
            if isinf(a)
                sig2(i) = 0.5*sig2(i);
            else
                sig2(i) = 0.5*(a + sig2(i));
            end
        end
        iter = iter + 1;
        if iter == maxit
            notConverge(i)=true;
            break;
        end
    end
    if nargin < 3
         condProbMatX(i,:) = P_i;
    else
        % Return full matrix for 'barneshut' algorithm
        condProbMatX(i,knnidx(i,:)) = P_i;
    end
end
if any(notConverge)
    warning(message('stats:tsne:BinarySearchNotConverge'));
end
end


function [grad,probMatY] = tsneGradient(probMatX,Y)
% Compute gradient of t-SNE
N = size(Y,1);
Ysum = sum(Y.^2,2);
numeratorProbMatY = 1 ./ (1 + bsxfun(@plus,Ysum, bsxfun(@plus,Ysum', -2*(Y*Y')))); 
numeratorProbMatY(1:N+1:end) = 0;
probMatY = max(numeratorProbMatY./sum(numeratorProbMatY(:)),realmin(class(Y)));
pdiff = numeratorProbMatY.*(probMatX - probMatY);
grad = 4 * (diag(sum(pdiff,1))-pdiff) * Y; 
end

function [Y,loss,time] = tsneEmbedding(Y,probMatX,exaggeration,...
              learnrate,numprint,verbose,options,algorithm,theta,colidx,rowcnt,...
                                  tsnepiopt)

[N,Ydims] = size(Y);
% Initialization
Ychange = zeros(N,Ydims,'like',Y);
adpRatechange = ones(size(Y),'like',Y);
minRatechange = 0.01;
momentums = [0.5 0.8];
momentumChange = 250;
exaggerationStop = 250;
titleChangeIter = ceil(exaggerationStop/numprint)*numprint;
numprintcalls = 0;
% Adaptive learning rate in reference Jacobs (1988)
k = 0.15;
phi = 0.85;

% keep early exaggeration as input for CSB
if strcmpi(algorithm, 'tsnepi')
  computegrad( tsnepiopt.destroycsb );
  computegrad( tsnepiopt.buildcsb, uint32(colidx-1)', uint32(cumsum([0 rowcnt]))', ...
               probMatX' );
end

% Early exaggeration
probMatX = exaggeration * probMatX;
iter = 1;

% Check for OutputFcn
haveOutputFcn = ~isempty(options.OutputFcn);
stop = false;
if haveOutputFcn
    pval = options.OutputFcn;
    if iscell(pval) && all(cellfun(@(x) isa(x,'function_handle'),pval))
        OutputFcn = pval;
    elseif isa(pval,'function_handle')
        OutputFcn = {pval};
    elseif isempty(pval)
        OutputFcn = {};
    else
        error(message('stats:tsne:InvalidOutputFcn'))
    end
    
    optimValues = struct('iteration',[],'fval',[],'grad',[],'Y',[],...
                    'Exaggeration',exaggeration);
    stop = callOutputFcns(OutputFcn,optimValues,'init');
end

tic
while (iter<=options.MaxIter && ~stop)
    if iter == exaggerationStop
        probMatX = probMatX/exaggeration;
        exaggeration = 1;
    end
    if strcmpi(algorithm,'exact')
        [grad,probMatY] = tsneGradient(probMatX,Y);
        if ( rem(iter,numprint) == 0 )
            entropyX = probMatX(:)'*log(probMatX(:)); 
            entropyY = probMatX(:)'*log(probMatY(:));
            loss = entropyX - entropyY;
        end
        
    elseif strcmpi(algorithm,'tsnepi')

        [dy,Z] = computegrad( tsnepiopt.computegrad, Y', exaggeration );
        grad = dy';
        
        % using built-in loss function
        if ( rem(iter,numprint) == 0 )
            loss = tsnelossmex(Y',colidx,rowcnt,probMatX,Z);
        end
        
    else
        % Compute gradient by Barnes-Hut algorithm
        ymin = min(Y,[],1);
        ymax = max(Y,[],1);
        ycenter = mean(Y,1);
        ywidth = max(ymax-ycenter,ycenter-ymin)+sqrt(eps(class(Y)));
        if isempty(colidx) || isempty(rowcnt) || isempty(probMatX)
            % Empty joint probability matrix
            grad = zeros(size(Y),'like',Y);
            loss = cast(0,'like',Y);
        else
            [attrForce,repForce,Z] = tsnebhmex(theta,Y',ycenter,ywidth,colidx,rowcnt,probMatX);
            grad = 4*(attrForce-repForce)';
            if ( rem(iter,numprint) == 0 )
                loss = tsnelossmex(Y',colidx,rowcnt,probMatX,Z);
            end
        end
    end
    
    % Adaptive learning rate
    opsIdx = sign(grad) ~= sign(Ychange);
    adpRatechange(opsIdx) = adpRatechange(opsIdx) + k;
    adpRatechange(~opsIdx) = adpRatechange(~opsIdx) * phi;
    adpLearnrate = learnrate * max(minRatechange,adpRatechange);
    
    % Gradient update
    if iter < momentumChange
        Ychange = momentums(1)*Ychange - adpLearnrate.*grad;
    else
        Ychange = momentums(2)*Ychange - adpLearnrate.*grad;
    end
    
    % Update Y
    Y = Y + Ychange;
        
    % Convergency information
    infnormg = norm(grad,Inf);
    if infnormg < options.TolFun
        if verbose >=1
            fprintf('%s\n',getString(message('stats:tsne:TerminatedNormOfGradient')));
        end
        break;
    end

    % Display convergence information
    if ( rem(iter,numprint) == 0 )
        % Perform outputfcn
        if haveOutputFcn
            % We only care about the states init, iter and done
            optimValues = struct('iteration',iter,'fval',loss,'grad',grad,'Y',Y,...
                            'Exaggeration',exaggeration);
            if iter<options.MaxIter+numprint
                stop = callOutputFcns(OutputFcn,optimValues,'iter');
            else
                stop = callOutputFcns(OutputFcn,optimValues,'done');
            end
        end
        if verbose>=1
            displayConvergenceInfo(iter,loss,infnormg,numprintcalls,exaggeration,titleChangeIter);
            numprintcalls = numprintcalls + 1;
        end
    end
    iter = iter + 1;
end

time = toc;

% cleanup CSB
computegrad( tsnepiopt.destroycsb );

  
if nargout>1
    % Compute the loss of the final step
    if strcmpi(algorithm,'exact')
        entropyX = probMatX(:)'*log(probMatX(:));
        entropyY = probMatX(:)'*log(probMatY(:));
        loss = entropyX - entropyY;
    else
        if isempty(colidx) || isempty(rowcnt) || isempty(probMatX)
            % Empty joint probability matrix
            loss = 0;
        else
            loss = tsnelossmex(Y',colidx,rowcnt,probMatX,Z);
        end
    end
end
end


function displayConvergenceInfo(iter,loss,infnormg,numprintcalls,exaggeration,titleChangeIter)
% Helper function to display iteration convergence info.

% |==============================================|
% |   ITER   |  KL DIVERGENCE  |     NORM GRAD   |
% |          |    FUN VALUE    |                 |
% |==============================================|
% |       20 |    1.211293e-01 |    8.905909e-05 |
% |       40 |    1.211192e-01 |    2.639418e-05 |
% |       60 |    1.211093e-01 |    3.889625e-05 |
% |       80 |    1.211076e-01 |    1.954810e-04 |
% |      100 |    1.210898e-01 |    5.100216e-05 |
% |      120 |    1.210793e-01 |    1.782637e-05 |
% |      140 |    1.210700e-01 |    4.753843e-05 |
% |      160 |    1.210612e-01 |    6.728603e-05 |
% |      180 |    1.210532e-01 |    8.736254e-05 |
% 

if iter<titleChangeIter && exaggeration>1
    if rem(numprintcalls,20) == 0
        fprintf('\n');
        fprintf('|==============================================|\n');
        fprintf('|   ITER   | KL DIVERGENCE   | NORM GRAD USING |\n');
        fprintf('|          | FUN VALUE USING | EXAGGERATED DIST|\n');
        fprintf('|          | EXAGGERATED DIST| OF X            |\n');
        fprintf('|          | OF X            |                 |\n');
        fprintf('|==============================================|\n');
    end
else
    % Title change
    if exaggeration==1 && iter==titleChangeIter && rem(numprintcalls,20) ~= 0
        fprintf('\n');
        fprintf('|==============================================|\n');
        fprintf('|   ITER   |  KL DIVERGENCE  |    NORM GRAD    |\n');
        fprintf('|          |    FUN VALUE    |                 |\n');
        fprintf('|==============================================|\n');     
    end
    if rem(numprintcalls,20) == 0
        fprintf('\n');
        fprintf('|==============================================|\n');
        fprintf('|   ITER   |  KL DIVERGENCE  |    NORM GRAD    |\n');
        fprintf('|          |    FUN VALUE    |                 |\n');
        fprintf('|==============================================|\n');
    end
end

fprintf('|%9d |%16.6e |%16.6e |\n', iter,loss,infnormg);

end

function stop = callOutputFcns(outputFcn,optimValues,state)
% Call each output function
stop = false;
for i = 1:numel(outputFcn)
    stop = stop | outputFcn{i}(optimValues,state);
end
end

function [colidx, rowcnt, probvec] = probMatXknn(probMatX,knnidx)
% Find joint probability matrix of nearest neighbors
% Use sparse matrix to save memory
[N,K] = size(probMatX);
if K<N
    SProwidx = bsxfun(@times,ones(K,N),(1:N));
    SProwidx = SProwidx(:);
    knnidx = knnidx';
    knnidx = knnidx(:);
    probMatX = probMatX';
    probMatX = probMatX(:);
    S = sparse(SProwidx,knnidx,probMatX,N,N);
    P = S+S';
else
    P = probMatX + probMatX';   
end
[rowidx,colidx,probvec] = find(P);
[rowidx,sridx] = sort(rowidx);
colidx = colidx(sridx)';
probvec = probvec(sridx)'./(2*N);
rowcnt = grpstats(rowidx,rowidx,'numel')';
end
