%% Parameters
numDevices = 4;          % UPDATED: 2 legitimate, 2 rogue
numEnrollSamples = 100;
numTestSamples = 100;   
SNR_dB = 30;             % was 20, cleaner signal so hardware fingerprint dominates
outputPath = '\\wsl.localhost\Ubuntu\home\bdeloatch\pla_data\features.csv';

%% Channel Configuration
channel = nrTDLChannel;
channel.DelayProfile = 'TDL-C';
channel.DelaySpread = 30e-9;    % was 100e-9, reduce multipath blur
channel.MaximumDopplerShift = 5;
channel.SampleRate = 15.36e6;
channel.NumTransmitAntennas = 1;
channel.NumReceiveAntennas = 1;

%% Device Hardware Profiles — UPDATED
% Each device has a mean hardware fingerprint + per-sample Gaussian drift

% Legitimate UE 1 — stable, well-calibrated radio
devices(1).label = 1;
devices(1).cfo_mean = 150;          % low CFO, well-calibrated oscillator
devices(1).cfo_std = 10;            % tight drift — stable hardware
devices(1).iq_mean = 0.020;         % minimal I/Q imbalance
devices(1).iq_std = 0.003;
devices(1).phase_mean = 0.010;      % low phase noise
devices(1).phase_std = 0.002;
devices(1).tdl_spread = 28e-9;      % short delay spread — close to gNB

% Legitimate UE 2 — slightly different hardware, similar location
devices(2).label = 1;
devices(2).cfo_mean = 200;          % moderate CFO, still within legitimate range
devices(2).cfo_std = 12;
devices(2).iq_mean = 0.030;         % slightly higher imbalance than Device 1
devices(2).iq_std = 0.004;
devices(2).phase_mean = 0.015;
devices(2).phase_std = 0.002;
devices(2).tdl_spread = 30e-9;

% Rogue UE 1 — intentionally close to Legitimate UE 2 in feature space
% This is the hard classification case — stress tests SVM decision boundary
devices(3).label = 0;
devices(3).cfo_mean = 230;          % overlaps with Device 2's CFO distribution
devices(3).cfo_std = 14;            % wider drift — less stable hardware
devices(3).iq_mean = 0.035;         % slightly higher than Device 2, borderline
devices(3).iq_std = 0.004;
devices(3).phase_mean = 0.018;      % close to Device 2's phase profile
devices(3).phase_std = 0.002;
devices(3).tdl_spread = 29e-9;      % similar environment to Device 2

% Rogue UE 2 — clearly different hardware, easier to classify
% Represents an unsophisticated attacker with poor radio hardware
devices(4).label = 0;
devices(4).cfo_mean = 270;          % noticeably higher CFO than any legitimate device
devices(4).cfo_std = 16;
devices(4).iq_mean = 0.055;         % high I/Q imbalance — poor hardware quality
devices(4).iq_std = 0.006;
devices(4).phase_mean = 0.028;      % elevated phase noise
devices(4).phase_std = 0.004;
devices(4).tdl_spread = 31e-9;

%% Feature Extraction Loop — UPDATED with per-sample hardware drift
featureMatrix = [];
labelVector = [];
deviceIDVector = [];

for d = 1:numDevices
    % Update channel delay spread per device — UPDATED
    release(channel);
    channel.DelaySpread = devices(d).tdl_spread;
    
    totalSamples = numEnrollSamples + numTestSamples;
    
    for s = 1:totalSamples
        %% Generate SRS-like OFDM waveform
        nFFT = 1024;
        nSubcarriers = 72;
        symbolsPerSlot = 14;
        
        txSymbols = (2*(rand(nSubcarriers,1)>0.5)-1) + ...
                    1j*(2*(rand(nSubcarriers,1)>0.5)-1);
        txSymbols = txSymbols / sqrt(2);
        
        txGrid = zeros(nFFT, symbolsPerSlot);
        startSC = (nFFT - nSubcarriers)/2 + 1;
        txGrid(startSC:startSC+nSubcarriers-1, end) = txSymbols;
        
        txWaveform = ifft(txGrid, nFFT);
        txWaveform = txWaveform(:);
        
        %% Apply per-sample hardware imperfections — UPDATED
        % CFO drifts slightly each transmission
        cfo_sample = devices(d).cfo_mean + devices(d).cfo_std * randn();
        iq_sample  = devices(d).iq_mean  + devices(d).iq_std  * randn();
        ph_sample  = devices(d).phase_mean + devices(d).phase_std * randn();
        
        % CFO
        t = (0:length(txWaveform)-1)' / channel.SampleRate;
        txWaveform = txWaveform .* exp(1j*2*pi*cfo_sample*t);
        
        % I/Q imbalance
        txWaveform = real(txWaveform)*(1 + iq_sample) + ...
                     1j*imag(txWaveform)*(1 - iq_sample);
        
        % Phase noise
        phNoise = ph_sample * randn(size(txWaveform));
        txWaveform = txWaveform .* exp(1j*phNoise);
        
        %% Pass through TDL channel
        reset(channel);
        [rxWaveform, ~] = channel(txWaveform);
        rxWaveform = awgn(rxWaveform, SNR_dB, 'measured');
        
        %% Feature Extraction (PRE-equalization)
        rxGrid = fft(reshape(rxWaveform(1:nFFT*symbolsPerSlot), ...
                    nFFT, symbolsPerSlot), nFFT);
        rxSymbols = rxGrid(startSC:startSC+nSubcarriers-1, end);
        
        % Feature 1: Estimated CFO
        phaseDiff = angle(rxSymbols(2:end) .* conj(rxSymbols(1:end-1)));
        estCFO = mean(phaseDiff) * channel.SampleRate / (2*pi);
        
        % Feature 2: I/Q imbalance estimate
        iComponent = real(rxSymbols);
        qComponent = imag(rxSymbols);
        estIQImbalance = std(iComponent) / std(qComponent);
        
        % Feature 3: Received signal power
        rxPower = mean(abs(rxSymbols).^2);
        
        % Feature 4: Phase variance
        phaseVar = var(angle(rxSymbols));
        
        % Feature 5: Amplitude variance
        ampVar = var(abs(rxSymbols));
        
        % UPDATED: Feature 6 — temporal CFO variance across subcarriers
        % Captures frequency-selective CFO drift unique to each device
        cfoVar = var(phaseDiff) * channel.SampleRate / (2*pi);
        
        featureRow = [estCFO, estIQImbalance, rxPower, phaseVar, ampVar, cfoVar];
        featureMatrix = [featureMatrix; featureRow];
        labelVector = [labelVector; devices(d).label];
        deviceIDVector = [deviceIDVector; d];
    end
end

%% Write to CSV — UPDATED: includes DeviceID column for per-device analysis
outputTable = array2table([featureMatrix, labelVector, deviceIDVector], ...
    'VariableNames', {'CFO','IQImbalance','RxPower','PhaseVar','AmpVar','CFOVar','Label','DeviceID'});

writetable(outputTable, outputPath);
fprintf('Feature extraction complete. %d samples written to %s\n', ...
    size(featureMatrix,1), outputPath);