%
% SCRIPT: TEST_QQ_ERROR
%
%   Test error for QQ
%


%% CLEAN-UP

clear
close all


%% PARAMETERS

n = 8000;
d = 3;
w = 20;
h = logspace(log10(0.4),log10(2.5),30);

errFunAbs = @(x,y) max( abs( x(:) - y(:) ) );
errFunRel = @(x,y) max( abs( (x(:) - y(:)) ./ x(:) ) );

switch d
  case 1
    et = 10^(-5.5548) .* h.^(2.7899);
  case 2
    et = 10^(-5.7319) .* h.^(2.7408);
  case 3
    et = 10^(-5.9056) .* h.^(2.2729);
end


%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);

%% GENERATE RANDOM UNIFORM DATA

fprintf( '...generate random uniform data...\n' ); 

y = 2*w*(rand(n, d) - 0.5);

fprintf( '   - DONE\n');

%% EXACT QQ

fprintf( '...exact QQ...\n' ); 

[Fg, Zg] = qq_exact( y );

fprintf( '   - DONE\n');


%% APPROXIMATE QQ

fprintf( '...approximate QQ...\n' ); 

for ih = 1:length(h)

  fprintf( '   - h = %.2f\n', h(ih) )
  
  [F, Z] = computegrad( 4, y', 0, h(ih) );
  F = F';

  ef(ih) = errFunAbs( Fg, F );
  ez(ih) = errFunAbs( Zg, Z );
  ea(ih) = approximateError(h(ih), 4, d, n);
  
end

mp = [ones(length(h),1) log10(h')] \ log10(ef');

em = 10^(mp(1)) .* h.^(mp(2));

fprintf( '   - DONE\n');


%% SHOW ERRORS

fprintf( '...show errors...\n' ); 

figure
loglog( h, [ef; ez; ea; em; et], 'x--' );
ylabel('||F - F_g||_{oo}')
xlabel('h (grid cell side length)')

legend( {'Absolute force error', ...
         'Zeta (normalization term) error', ...
         'Theoretical bound', ...
         'Fitted model', ...
         'Error formula'} )

fprintf( '   - DONE\n');



%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);


function e = approximateError(h, p, d, n)

  switch d
    
    case 1
      
      e = 7 * (p+2)/p * (2^p * (h*p)^p) / exp(p);
      
    case 2
      
      e = 16 * 3 * (p+2)/(sqrt(8)*p^3) * (8^p * (h*p)^p) / exp(p);
      
    otherwise
      e = NaN;
      
  end

end


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION       0.1
%
% TIMESTAMP     <Sep 23, 2019: 15:05:40 Dimitris>
%
% ------------------------------------------------------------

