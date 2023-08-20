function Feature = Feature_Extract(EEG,kernel,type)

bandname = {'DELTA','THETA','ALPHA','BETA','GAMMA','all'};
if strcmp(type,'source')
    sourceflg = 1;
else 
    sourceflg = 0;
end
cfg.inverseMethod = 'MNE';
cfg.epochSize = 5;
all_powspctrm = [];
all_freq = [];
all_coh = [];
all_wpli_debiased = [];
%% EEG trials
if EEG.trials == 1
    eppos = 1:EEG.srate*cfg.epochSize:size(EEG.data,2);
    for ep = 1:length(eppos)
        EEG.event(ep).type = 'epo';
        EEG.event(ep).latency = eppos(ep);
    end
    EEG.event(length(eppos)+1:end) = [];
    EEG = pop_epoch(EEG,{'epo'},[0 cfg.epochSize]);
end

%% computing kernel
if sourceflg == 1
    ds = 3;
    sourceData = kernel*squeeze(mean(EEG.data, 3))*1e-6;
    transmatrix = zeros((size(sourceData,1)/ds), size(sourceData,1));
    for ii = 1: size(sourceData,1)/ds
        sourcetemp = sourceData((ii-1)*ds+1:ii*ds,:);
        R = sourcetemp*sourcetemp';
        [VV, DD] = eig(R);
        [~, I] = sort(diag(DD),'descend');
        transmatrix(ii,(ii-1)*ds+1:ii*ds) = VV(:,I(1))';
    end
    cfg.PCkernel = transmatrix*kernel;
else 
    cfg.PCkernel = kernel;
end
%% Feature calculation
for jj = 1:length(bandname)-1
    disp(['======' 'Band ' bandname{jj} ' Analysis' '======']);
    %% Feature Calculation Parameters      
    cfg.bandname = bandname{jj};
    switch cfg.bandname
        case 'DELTA'
            cfg.foi = 1:0.5:3;
        case 'THETA'
            cfg.foi = 4:0.5:7;
        case 'ALPHA'
            cfg.foi = 8:0.5:12;
        case 'BETA'
            cfg.foi = 13:0.5:30;
        case 'GAMMA'
            cfg.foi = 31:0.5:50;
    end

    %% Feature calculation
    cfg.FeMethod         = 'coh';
    cfgcoh               = [];
    cfgcoh.method        = 'mtmfft';
    cfgcoh.pad           = 'nextpow2';
    cfgcoh.taper         = 'dpss';
    cfgcoh.output        = 'fourier';     % 'fourier', 'powandcsd'
    cfgcoh.foi           = cfg.foi;
    cfgcoh.channel       = 'all';
    cfgcoh.keeptrials    =  'yes';
    cfgcoh.tapsmofrq     = 1.9;
    cfgcc.method         = cfg.FeMethod;
    cfgcc.complex        = 'imag';  %'abs' (default), 'angle', 'complex', 'imag', 'real'

    % compute features at channel space and then project them to source space
    % compute spectrum
    Data = eeglab2fieldtrip(EEG,'preprocessing','chanlocs');
    powcsd = ft_freqanalysis(cfgcoh,Data);
    temppw=powcsd.fourierspctrm;
    % source orientation
    for ii=1:size(temppw,3)
        pwspect(:,:,ii)=temppw(:,:,ii)*cfg.PCkernel';
    end
    powcsd.fourierspctrm=pwspect;
    for lb=1:size(cfg.PCkernel,1) powcsd.label{1,lb}=num2str(lb); end
    clear pwspect temppw;
    tempcoh = ft_connectivityanalysis(cfgcc,powcsd);
    Feature.(bandname{jj}).coh = mean(tempcoh.cohspctrm,3);
    Feature.(bandname{jj}).coh = single(Feature.(bandname{jj}).coh);
    all_coh = cat(3,all_coh,tempcoh.cohspctrm(:,:,1:2:end));
    clear powcsd tempcoh

end
 

end


