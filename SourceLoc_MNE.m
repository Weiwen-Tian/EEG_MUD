function Kernel = SourceLoc_MNE(Cw, HeadModel, OPTIONS)
% SourceLoc_MNE: MNE source localization

if (OPTIONS.UseDepth && strcmpi(OPTIONS.InverseMeasure, 'sLORETA'))
    disp('Depth weighting is not necessary when using sLORETA normalization, ignoring option UseDepth=1');
    OPTIONS.UseDepth = 0;
end

iW_noise = zeros(size(Cw));
Cw = double((Cw + Cw')/2);

[~,iW_noise] = truncate_and_regularize_covariance(Cw, OPTIONS.NoiseMethod, OPTIONS.NoiseReg);

NumDipoles = size(HeadModel.GridLoc,1);

Wq = cell(1,NumDipoles);
Alpha = ones(1,NumDipoles); % initialize to unity

if OPTIONS.UseDepth
    % See eq. 6.2.10 of MNE Manual version 2.7 (Mar 2010).
    % Original code had some backflips to check for instabilities.
    % Here we take a simpler direct approach.
    
    % We are assuming unconstrained (three source directions) per source
    % point. We form the unconstrained norm of each point
    ColNorms2 = sum(HeadModel.Gain .* HeadModel.Gain); % square of each column
    SourceNorms2 = sum(reshape(ColNorms2,3,[]),1); % Frobenius norm squared of each source
    
    % Now Calculate the *non-inverted* value
    Alpha2 = SourceNorms2 .^ OPTIONS.WeightExp; % note not negative weight (yet)
    AlphaMax2 = max(Alpha2); % largest squared source norm
    % The limit is meant to keep the smallest from getting to small
    Alpha2 = max(Alpha2, AlphaMax2 ./ (OPTIONS.WeightLimit^2)); % sets lower bound on source strengths
    
    % Now invert it
    Alpha2 = AlphaMax2 ./ Alpha2; % goes from WeightLimit^2 to 1, depending on the inverse strength of the source
    
    Alpha = sqrt(Alpha2);
    % Thus deep weak sources can be amplified up to WeightLimit times greater
    % strength, relative to the stronger shallower sources.
end


for i = 1:NumDipoles
    
    switch OPTIONS.SourceOrient
        case 'fixed'
            % fprintf('BST_INVERSE > Using constrained surface orientations\n');
            NumDipoleComponents = 1;
            tmp = HeadModel.GridOrient(i,:)'; % 3 x 1
            Wq{i} = tmp/norm(tmp); % ensure unity
            
        case 'loose'
            % fprintf('BST_INVERSE > Using loose surface orientations\n');
            NumDipoleComponents = 3;
            tmp = HeadModel.GridOrient(i,:)'; % preferred direction
            tmp = tmp/norm(tmp); % ensure unity
            tmp_perp = null(tmp'); % orientations perpedicular to preferred
            Wq{i} = [tmp tmp_perp*OPTIONS.Loose]; % weaken the other directions
            
        case 'free'
            % fprintf('BST_INVERSE > Using unconstrained orientations\n');
            NumDipoleComponents = 3;
            Wq{i} = eye(NumDipoleComponents);
            
        otherwise
            error('Unknown Source Orientation')
    end
    
    % L2norm of Wq in everycase above is 1, (even for loose, since L2 norm
    % of matrix is largest singular value of the matrix).
    
    Wq{i} = Alpha(i)*Wq{i}; % now weight by desired amplifier
    
    % So L2 norm of Wq is equal to the desired amplifier (if any).
end



% put all covariance priors into one big sparese matrix
WQ = blkdiag(sparse(Wq{1}),Wq{2:end});
% (by making first element sparse, we force Matlab to use efficient sparse
% mex function)

% With the above defined, then the whitened lead field matrix is simply

L = iW_noise * (HeadModel.Gain * WQ);  % if LCMV, this is data whitened.

[UL,SL2] = svd(L*L');
SL2 = diag(SL2);
SL = sqrt(SL2);

switch (OPTIONS.SnrMethod)
    case 'rms'
        % user had directly specified the variance
        Lambda =  OPTIONS.SnrRms^2;
        SNR = Lambda * SL2(1); % the assumed SNR for the entire leadfield
    case 'fixed'
        % user had given a desired SNR, set lambda of the Grammian to achieve it
        SNR = OPTIONS.SnrFixed^2;
        
        % several options here. Hamalainen prefers the average eigenvalue
        % of the Grammian. Mosher would prefer the maximum (norm) for other
        % consistency measures, however user's have become accustomed to
        % the Hamalainen measure.
        
        % Maximum (matrix norm):
        % Lambda = SNR/(SL(1)^2); % thus SL2(1)*Lambda = desired SNR.
        
        % Hamalainen definition of SNR in the min norm:
        Lambda = SNR/mean(SL.^2); % should be equalivent to Matti's average eigenvalue definition
        
    otherwise
        error(['Not supported yet: NoiseMethod="' OPTIONS.SnrMethod '"']);
end


switch OPTIONS.InverseMeasure
    case 'amplitude'
        Kernel = Lambda * L' * (UL * diag(1./(Lambda * SL2 + 1)) * UL')*iW_noise;
    case 'dSPM'
        Kernel = Lambda * L' * (UL * diag(1./(Lambda * SL2 + 1)) * UL');
        dspmdiag = sum(Kernel .^2, 2);
        if (NumDipoleComponents == 1)
            dspmdiag = sqrt(dspmdiag);
        elseif (NumDipoleComponents==3 || NumDipoleComponents==2)
            dspmdiag = reshape(dspmdiag, NumDipoleComponents,[]);
            dspmdiag = sqrt(sum(dspmdiag,1)); % Taking trace and sqrt.
            dspmdiag = repmat(dspmdiag, [NumDipoleComponents, 1]);
            dspmdiag = dspmdiag(:);
        end
        Kernel = Kernel ./repmat(dspmdiag, 1, size(Kernel, 2));
        Kernel = Kernel * iW_noise; % overall whitener
    case 'sLORETA'
        % calculate the standard min norm solution
        Kernel = Lambda * L' * (UL * diag(1./(Lambda * SL2 + 1)) * UL');
        
        if (NumDipoleComponents == 1)
            sloretadiag = sqrt(sum(Kernel .* L', 2));
            Kernel = Kernel ./repmat(sloretadiag, 1, size(Kernel, 2));
        elseif (NumDipoleComponents==3 || NumDipoleComponents==2)
            for spoint = 1:NumDipoleComponents:size(Kernel,1),
                R = Kernel(spoint:spoint+NumDipoleComponents-1,:) * L(:,spoint:spoint+NumDipoleComponents-1);
                % SIR = sqrtm(pinv(R)); % Aug 2016 can lead to errors if
                % singular Use this more explicit form instead
                [Ur,Sr,Vr] = svd(R); Sr = diag(Sr);
                sumSr = 0;
                for n = 1:length(Sr)
                    sumSr = sumSr + Sr(n);
                    if sumSr/sum(Sr) >= 0.999
                        break;
                    end
                end
                RNK = n;
                
                
                SIR = Vr(:,1:RNK) * diag(1./sqrt(Sr(1:RNK))) * Ur(:,1:RNK)'; % square root of inverse
                
                Kernel(spoint:spoint+NumDipoleComponents-1,:) = SIR * Kernel(spoint:spoint+NumDipoleComponents-1,:);
            end
        end
        
        Kernel = Kernel * iW_noise; % overall whitener
end

if strcmpi(OPTIONS.InverseMeasure, 'amplitude')
    % we need to put orientation and weighting back into solution
    if NumDipoleComponents == 3 % we are in three-d,
        Kernel = WQ * double(Kernel); % put the source prior back into the solution
    elseif NumDipoleComponents == 1
        Kernel = diag(Alpha)*double(Kernel);  % put possible alpha weighting back in
    end
end




end
%% =========== Covariance Truncation and Regularization
function [Cov,iW] = truncate_and_regularize_covariance(Cov,Method,NoiseReg)
% Cov is the covariance matrix, to be regularized using Method
% Type is the sensor type for display purposes
% NoiseReg is the regularization fraction, if Method "reg" selected
% FourthMoment and nSamples are used if Method "shrinkage" selected

VERBOSE = true; % be talkative about what's happening

% Ensure symmetry
Cov = (Cov + Cov')/2;

% Note,impossible to be complex by above symmetry check
% Decompose just this covariance.
[Un,Sn2] = svd(Cov,'econ');
Sn = sqrt(diag(Sn2)); % singular values

sumSn = 0;
for n = 1:length(Sn)
    sumSn = sumSn + Sn(n);
    if sumSn/sum(Sn) >= 0.999
        break;
    end
end


Rank_Noise = n;


Un = Un(:,1:Rank_Noise);
Sn = Sn(1:Rank_Noise);

% now rebuild the noise covariance matrix with just the non-zero
% components
Cov = Un*diag(Sn.^2)*Un'; % possibly deficient matrix now

% With this modality truncated, see if we need any additional
% regularizations, and build the inverse whitener

if VERBOSE
    fprintf('Using the ''%s'' method of covariance regularization.\n',Method);
end

switch(Method) % {'shrink', 'reg', 'diag', 'none', 'median'}
    
    case 'none'
        %  "none" in Regularization means no
        % regularization was applied to the computed Noise Covariance
        % Matrix
        % Do Nothing to Cw_noise
        iW = Un*diag(1./Sn)*Un'; % inverse whitener
        if VERBOSE,
            fprintf('No regularization applied to covariance matrix.\n');
        end
        
        
    case 'median'
        if VERBOSE,
            fprintf('Covariance regularized by flattening tail of eigenvalues spectrum to the median value of %.1e\n',median(Sn));
        end
        Sn = max(Sn,median(Sn)); % removes deficient small values
        Cov = Un*diag(Sn.^2)*Un'; % rebuild again.
        iW = Un*diag(1./Sn)*Un'; % inverse whitener
        
    case 'diag'
        Cov = diag(diag(Cov)); % strip to diagonal
        iW = diag(1./sqrt(diag(Cov))); % inverse of diagonal
        if VERBOSE,
            fprintf('Covariance matrix reduced to diagonal.\n');
        end
        
    case 'reg'
        % The unit of "Regularize Noise Covariance" is as a percentage of
        % the mean variance of the modality.
        
        % Ridge Regression:
        RidgeFactor = Sn2(1) * NoiseReg ; % percentage of max
        
        Cov = Cov + RidgeFactor * eye(size(Cov,1));
        iW = Un*diag(1./(Sn + sqrt(RidgeFactor)))*Un'; % inverse whitener
        
        if VERBOSE,
            fprintf('Diagonal of %.1f%% of largest eigenvalue added to covariance matrix.\n',NoiseReg * 100);
        end
        
    otherwise
        error(['Unknown covariance regularization method: NoiseMethod="' Method '"']);
        
end % method of regularization


% Note the design of full rotating whiteners. We don't expect dramatic reductions in
% rank here, and it's convenient to rotate back to the original space.
% Note that these whitener matrices may not be of full rank.

end