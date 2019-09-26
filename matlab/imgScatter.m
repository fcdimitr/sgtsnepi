function G = imgScatter(x, Img, szFullImg, szSepImg)
% IMGSCATTER - Draw scatter plot with images
%   
% IMGSACTTER( COORD, IMG ) displays a 2D image with images defined in
% variable IMG at 2-D coordinates in matrix COORD.
% 
% Basic code by Andrej Karpathy
% https://cs.stanford.edu/people/karpathy/cnnembed/
% 
% Minor modifications and function creation by Dimitris Floros

if ~exist( 'szFullImg', 'var' ) || isempty( szFullImg )
  szFullImg = 2000;
end 

if ~exist( 'szSepImg', 'var' ) || isempty( szSepImg )
  szSepImg = 50;
end 


%% load embedding

x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

%% load validation image filenames

N = size(Img, 4);

%% create an embedding image

S = szFullImg; % size of full embedding image
G = 255*ones(S, S, 3, 'uint8');
s = szSepImg; % size of every single image

for i=1:N
    
    if mod(i, 10000)==0
        fprintf('%d/%d...\n', i, N);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 255
        continue % spot already filled
    end
    
    I = Img(:,:,:,i);
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

imshow(G);

%%
%imwrite(G, 'cnn_embed_2k.jpg', 'jpg');

%% average up images
% % (doesnt look very good, failed experiment...)
% 
% S = 1000;
% G = zeros(S, S, 3);
% C = zeros(S, S, 3);
% s = 50;
% 
% Ntake = 5000;
% for i=1:Ntake
%     
%     if mod(i, 100)==0
%         fprintf('%d/%d...\n', i, Ntake);
%     end
%     
%     % location
%     a = ceil(x(i, 1) * (S-s-1)+1);
%     b = ceil(x(i, 2) * (S-s-1)+1);
%     a = a-mod(a-1,s)+1;
%     b = b-mod(b-1,s)+1;
%     
%     I = imread(fs{i});
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
%     
%     G(a:a+s-1, b:b+s-1, :) = G(a:a+s-1, b:b+s-1, :) + double(I);
%     C(a:a+s-1, b:b+s-1, :) = C(a:a+s-1, b:b+s-1, :) + 1;
%     
% end
% 
% G(C>0) = G(C>0) ./ C(C>0);
% G = uint8(G);
% imshow(G);

%% do a guaranteed quade grid layout by taking nearest neighbor

% S = 2000; % size of final image
% G = zeros(S, S, 3, 'uint8');
% s = 50; % size of every image thumbnail

% xnum = S/s;
% ynum = S/s;
% used = false(N, 1);

% qq=length(1:s:S);
% abes = zeros(qq*2,2);
% i=1;
% for a=1:s:S
%     for b=1:s:S
%         abes(i,:) = [a,b];
%         i=i+1;
%     end
% end
% %abes = abes(randperm(size(abes,1)),:); % randperm

% for i=1:size(abes,1)
%     a = abes(i,1);
%     b = abes(i,2);
%     %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
%     %yf = ((b-1)/S - 0.5)/2 + 0.5;
%     xf = (a-1)/S;
%     yf = (b-1)/S;
%     dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
%     dd(used) = inf; % dont pick these
%     [dv,di] = min(dd); % find nearest image

%     used(di) = true; % mark as done
%     I = Img(:,:,:,i);
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);

%     G(a:a+s-1, b:b+s-1, :) = I;

%     if mod(i,100)==0
%         fprintf('%d/%d\n', i, size(abes,1));
%     end
% end

% imshow(G);

%%
% imwrite(G, 'cnn_embed_full_2k.jpg', 'jpg');
