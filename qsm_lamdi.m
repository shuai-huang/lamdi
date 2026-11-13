function [chi, result] = qsm_lamdi(opts)
    % QSM_LAMDI Reconstruct a susceptibility map using LAMDI.
    %
    %   [CHI, RESULT] = QSM_LAMDI(OPTS) runs Laplace-Mixture Dipole Inversion
    %   (LAMDI), an AMP-PE dipole inversion method with a two-component Laplace
    %   mixture prior on 3-D wavelet coefficients.
    %
    %   This version assumes that the input phase/local-field image, brain mask,
    %   and magnitude image are stored in MATLAB arrays or in a MATLAB
    %   .mat file.
    %
    %   Required input options:
    %       opts.dataFile          MAT file containing the input arrays. 
    %
    %   Variable names in opts.dataFile:
    %       opts.phaseVariable     variable (name) containing the phase/local field;
    %                              default: 'phase'
    %                              It contains 3-D/4-D phase or local-field array.
    %       opts.maskVariable      variable (name) containing the brain mask;
    %                              default: 'mask'
    %                              It contains 3-D brain mask
    %       opts.magnitudeVariable variable (name) containing the magnitude image;
    %                              default: 'magnitude'. If this variable is not
    %                              found, unit weights are used.
    %                              It contains 3-D/4-D magnitude image
    %
    %   Important optional fields:
    %       opts.fieldMapUnits     'radian' (default), or 'ppm'
    %       opts.echoTimes         echo time(s), in seconds; default: 8e-3
    %       opts.fieldStrengthT    main field strength, in tesla; default: 3
    %       opts.voxelSize         voxel size, in mm; default: [1 1 1]
    %       opts.waveletName       wavelet name; default: 'db1'
    %       opts.waveletLevel      decomposition level; default: 3
    %
    %   External dependencies expected on the MATLAB path include the Wavelet
    %   Toolbox, the GAMP LinTrans base class, and the QSM utility functions used
    %   by the original pipeline: GenerateDipoleFT3Drot, SI_operator_withmask,
    %   dipole_kernel_angulated, and chiL2.
    %
    %   Example:
    %       opts = struct();
    %       opts.dataFile = 'subject01_lamdi_input.mat';
    %       opts.phaseVariable = 'phase';       
    %       opts.maskVariable = 'mask';
    %       opts.magnitudeVariable = 'magnitude';
    %       opts.fieldMapUnits = 'radian';  % phase in radian by default, change it to ppm if need be
    %       opts.echoTimes = 8e-3;
    %       opts.voxelSize = [1 1 1];
    %       [chi, result] = qsm_lamdi(opts);

    if nargin < 1 || ~isstruct(opts)
        error('qsm_lamdi:InvalidInput', 'Input must be an options structure.');
    end

    opts = setDefaultOptions(opts);

    if ~isempty(opts.functionsDir)
        addpath(genpath(opts.functionsDir));
    end

    if ~isempty(opts.numThreads)
        maxNumCompThreads(opts.numThreads);
    end

    localFieldOrPhase = loadMatVariable(opts.dataFile, opts.phaseVariable);
    mask = loadMatVariable(opts.dataFile, opts.maskVariable);
    magnitude = loadMatVariable(opts.dataFile, opts.magnitudeVariable);

    if ndims(mask) ~= 3
        error('qsm_lamdi:InvalidMask', 'The brain mask must be a 3-D array.');
    end

    matSize = size(mask);
    if ~isequal(size(localFieldOrPhase, 1), matSize(1)) || ...
       ~isequal(size(localFieldOrPhase, 2), matSize(2)) || ...
       ~isequal(size(localFieldOrPhase, 3), matSize(3))
        error('qsm_lamdi:DimensionMismatch', ...
            'The local field/phase image and brain mask must have matching spatial dimensions.');
    end

    if ~isempty(opts.dwtMode)
        dwtmode(opts.dwtMode, 'nodisp');
    end

    nVox = nnz(mask);
    echoTimes = opts.echoTimes(:).';
    nEcho = numel(echoTimes);
    phaseScale = 2*pi*opts.gyromagnetic_ratio*opts.fieldStrengthT*echoTimes;

    [phaseImage, fieldPpm] = preparePhaseData(localFieldOrPhase, mask, opts, phaseScale);
    [measurementWeight, ~] = prepareMagnitudeWeights(magnitude, mask, nEcho, opts);

    rotMat = opts.rotationMatrix;
    if ~isequal(size(rotMat), [3 3])
        error('qsm_lamdi:InvalidRotationMatrix', 'opts.rotationMatrix must be 3-by-3.');
    end

    qsmOpt.plane = rotMat;
    voxelSize = opts.voxelSize;
    b0Dir = rotMat(3, :);

    % construct dipole kernel
    dipoleFT = GenerateDipoleFT3Drot(matSize, voxelSize, qsmOpt.plane);
    A = SI_operator_withmask(dipoleFT, matSize, mask);

    % L2 initialization used only to estimate an initial coefficient scale.
    kernel = dipole_kernel_angulated(matSize, opts.voxelSize, b0Dir);
    fieldVolume = zeros(matSize);
    fieldVolume(mask==1) = fieldPpm;
    chiInit = chiL2(fieldVolume, mask, kernel, opts.l2Beta, matSize);
    chiInit = real(chiInit);
    chiInit(mask==0) = 0;

    % construct wavelet transform and its inverse operator
    waveletTemplate = wavedec3(zeros(matSize), opts.waveletLevel, opts.waveletName);
    psi = @(x) wavedec3(x, opts.waveletLevel, opts.waveletName);
    psit = @(w) waverec3(w);
    coeffToStruct = @(coeff) buildWaveletStruct(coeff, waveletTemplate);
    structToCoeff = @(w) extractWaveletCoefficients(w);

    initCoeff = structToCoeff(psi(chiInit));
    nCoeff = numel(initCoeff);

    % Preserve the behavior of the original code: use the L2 solution to set the
    % initial prior scale, then start the nonlinear AMP iterations from zero.
    % DO NOT initialize AMP with the L2 solution because it is a bad initialization
    % Initialize AMP with an all-zero solution in the first linearized model
    chi = zeros(matSize);
    phaseEstimate = applyPhaseOperator(A, chi, phaseScale, nEcho);
    expPhaseEstimate = exp(1i * phaseEstimate);
    
    expPhaseImage = exp(1i * phaseImage);

    inputPar = initializeInputParameters(initCoeff, opts);
    outputPar.tau_w = opts.initialNoiseVariance;

    gampPar = struct();
    gampPar.damping_rate_alpha = opts.gampDampingAlpha;
    gampPar.damping_rate_beta = opts.gampDampingBeta;
    gampPar.max_pe_spar_ite = opts.gampMaxIterations;
    gampPar.max_pe_est_ite = opts.gampParameterIterations;
    gampPar.cvg_thd = opts.gampConvergenceTolerance;
    gampPar.verbose = opts.verbose;
    gampPar.x_hat_meas = zeros(nCoeff, 1);
    gampPar.tau_x_meas = repmat(max(var(initCoeff), eps), nCoeff, 1);
    gampPar.s_hat_meas = zeros(size(phaseImage));

    residueWeight = ones(size(phaseImage));

    history.nonlinearConvergence = nan(opts.maxOuterIterations, 1);
    history.residualNorm = nan(opts.maxOuterIterations, 1);
    history.lambda1 = nan(opts.maxOuterIterations, 1);
    history.lambda2 = nan(opts.maxOuterIterations, 1);
    history.mixtureWeight1 = nan(opts.maxOuterIterations, 1);
    history.mixtureWeight2 = nan(opts.maxOuterIterations, 1);
    history.tauW = nan(opts.maxOuterIterations, 1);

    lastRes = struct();
    for outerIter = 1:opts.maxOuterIterations
        
        firstDerivative = 1i * expPhaseEstimate;

        linWeight = firstDerivative .* measurementWeight .* residueWeight;
        A_lamdi = A_lamdi_LinTrans(nVox*nEcho, nCoeff, matSize, linWeight, A, ...
            phaseScale, psit, psi, coeffToStruct, structToCoeff, nVox, nEcho);

        linearizedMeasurement = measurementWeight .* residueWeight .* ...
            (firstDerivative .* phaseEstimate + expPhaseImage - expPhaseEstimate);

        [lastRes, inputPar, outputPar] = gamp_lap_mix(A_lamdi, linearizedMeasurement, ...
            gampPar, inputPar, outputPar);
        
        % update the initialization for the next round of AMP
        gampPar.x_hat_meas = lastRes.x_hat_meas;
        gampPar.tau_x_meas = lastRes.tau_x_meas;
        gampPar.s_hat_meas = lastRes.s_hat_meas;

        chiPrev = chi;
        chi = real(psit(coeffToStruct(lastRes.x_hat_meas)));
        
        % set the values outside the mask to zero
        %chi(mask==0) = 0;

        denom = max(norm(chi(:), 'fro'), eps);
        nonlinearCvg = norm(chi(:) - chiPrev(:), 'fro') / denom;

        phaseEstimate = applyPhaseOperator(A, chi, phaseScale, nEcho);
        expPhaseEstimate = exp(1i * phaseEstimate);
        
        residual = measurementWeight .* (expPhaseEstimate - expPhaseImage);
        residualNorm = norm(residual(:), 'fro');
        if (opts.useResidueWeight)
            residueWeight = updateResidualWeights(residual, opts.residualSigma);
        end

        history.nonlinearConvergence(outerIter) = nonlinearCvg;
        history.residualNorm(outerIter) = residualNorm;
        history.lambda1(outerIter) = inputPar.lambda_1;
        history.lambda2(outerIter) = inputPar.lambda_2;
        history.mixtureWeight1(outerIter) = inputPar.lap_w1;
        history.mixtureWeight2(outerIter) = inputPar.lap_w2;
        history.tauW(outerIter) = outputPar.tau_w;

        if opts.verbose
            fprintf(['LAMDI outer %03d: nonlinear_cvg=%0.3e, residual=%0.3e, ' ...
                     'lambda=(%0.3e,%0.3e), mix=(%0.3f,%0.3f), tau_w=%0.3e\n'], ...
                outerIter, nonlinearCvg, residualNorm, inputPar.lambda_1, inputPar.lambda_2, ...
                inputPar.lap_w1, inputPar.lap_w2, outputPar.tau_w);
        end

        if opts.outerConvergenceTolerance > 0 && nonlinearCvg < opts.outerConvergenceTolerance
            history.nonlinearConvergence = history.nonlinearConvergence(1:outerIter);
            history.residualNorm = history.residualNorm(1:outerIter);
            history.lambda1 = history.lambda1(1:outerIter);
            history.lambda2 = history.lambda2(1:outerIter);
            history.mixtureWeight1 = history.mixtureWeight1(1:outerIter);
            history.mixtureWeight2 = history.mixtureWeight2(1:outerIter);
            history.tauW = history.tauW(1:outerIter);
            break;
        end
    end

    result = struct();
    result.chi = chi;
    result.history = history;
    result.input_par = inputPar;
    result.output_par = outputPar;
    result.last_gamp = lastRes;
    result.residual_weight = residueWeight;
    result.wavelet_template = waveletTemplate;

end

function opts = setDefaultOptions(opts)
    opts = setDefault(opts, 'functionsDir', './functions');
    opts = setDefault(opts, 'dataFile', '');
    opts = setDefault(opts, 'phaseVariable', 'phase');
    opts = setDefault(opts, 'maskVariable', 'mask');
    opts = setDefault(opts, 'magnitudeVariable', 'magnitude');
    opts = setDefault(opts, 'fieldMapUnits', 'radian');
    opts = setDefault(opts, 'echoTimes', 8e-3);
    opts = setDefault(opts, 'fieldStrengthT', 3);
    opts = setDefault(opts, 'gyromagnetic_ratio', 42.58);
    opts = setDefault(opts, 'voxelSize', [1 1 1]);
    opts = setDefault(opts, 'rotationMatrix', eye(3));
    opts = setDefault(opts, 'waveletName', 'db1');
    opts = setDefault(opts, 'waveletLevel', 3);
    opts = setDefault(opts, 'dwtMode', 'per');
    opts = setDefault(opts, 'l2Beta', 2e-2);
    opts = setDefault(opts, 'maxOuterIterations', 100);
    opts = setDefault(opts, 'outerConvergenceTolerance', 1e-6);
    opts = setDefault(opts, 'residualSigma', 6);
    opts = setDefault(opts, 'initialNoiseVariance', 1e-12);
    opts = setDefault(opts, 'initialMixtureWeights', [0.9 0.1]);
    opts = setDefault(opts, 'initialSecondLambdaScale', 0.5);
    opts = setDefault(opts, 'gampDampingAlpha', 0.01);
    opts = setDefault(opts, 'gampDampingBeta', 0.1);
    opts = setDefault(opts, 'gampMaxIterations', 10);
    opts = setDefault(opts, 'gampParameterIterations', 1);
    opts = setDefault(opts, 'gampConvergenceTolerance', 1e-6);
    opts = setDefault(opts, 'useEchoSpecificMagnitudeWeights', true);
    opts = setDefault(opts, 'useResidueWeight', true);
    opts = setDefault(opts, 'numThreads', []);
    opts = setDefault(opts, 'verbose', true);

    if ~(hasNonemptyField(opts, 'dataFile'))
        error('qsm_lamdi:MissingPhase', ...
            'Provide opts.dataFile containing opts.phaseVariable.');
    end

    if hasNonemptyField(opts, 'dataFile')
        [~, ~, ext] = fileparts(opts.dataFile);
        if ~strcmpi(ext, '.mat')
            error('qsm_lamdi:InvalidDataFile', ...
                'opts.dataFile must be a MATLAB .mat file.');
        end
        if exist(opts.dataFile, 'file') ~= 2
            error('qsm_lamdi:DataFileNotFound', ...
                'Could not find opts.dataFile: %s', opts.dataFile);
        end
    end

    if ~isnumeric(opts.echoTimes) || isempty(opts.echoTimes) || ...
            any(~isfinite(opts.echoTimes(:))) || any(opts.echoTimes(:) <= 0)
        error('qsm_lamdi:InvalidEchoTimes', ...
            'opts.echoTimes must contain one or more positive echo times in seconds.');
    end
    opts.echoTimes = double(opts.echoTimes(:).');

    if ~isnumeric(opts.fieldStrengthT) || ~isscalar(opts.fieldStrengthT) || ...
            ~isfinite(opts.fieldStrengthT) || opts.fieldStrengthT <= 0
        error('qsm_lamdi:InvalidFieldStrength', ...
            'opts.fieldStrengthT must be a positive scalar in tesla.');
    end

    if ~isnumeric(opts.gyromagnetic_ratio) || ~isscalar(opts.gyromagnetic_ratio) || ...
            ~isfinite(opts.gyromagnetic_ratio) || opts.gyromagnetic_ratio <= 0
        error('qsm_lamdi:InvalidGyromagneticRatio', ...
            'opts.gyromagnetic_ratio must be a positive scalar in MHz/T.');
    end

    if numel(opts.voxelSize) ~= 3 || any(~isfinite(opts.voxelSize(:))) || ...
            any(opts.voxelSize(:) <= 0)
        error('qsm_lamdi:InvalidVoxelSize', ...
            'opts.voxelSize must contain three positive finite elements.');
    end
    opts.voxelSize = double(opts.voxelSize(:).');

    if numel(opts.initialMixtureWeights) ~= 2 || any(~isfinite(opts.initialMixtureWeights(:))) || ...
            any(opts.initialMixtureWeights(:) < 0) || sum(opts.initialMixtureWeights(:)) <= 0
        error('qsm_lamdi:InvalidMixtureWeights', ...
            'opts.initialMixtureWeights must contain two nonnegative finite values with positive sum.');
    end
    opts.initialMixtureWeights = double(opts.initialMixtureWeights(:).') / sum(opts.initialMixtureWeights(:));
end

function opts = setDefault(opts, fieldName, defaultValue)
    if ~isfield(opts, fieldName) || isempty(opts.(fieldName))
        opts.(fieldName) = defaultValue;
    end
end

function tf = hasNonemptyField(s, fieldName)
    tf = isfield(s, fieldName) && ~isempty(s.(fieldName));
end

function [x, found] = loadMatVariable(filename, variableName)
    %LOADMATVARIABLE Load one variable from a MAT file without reading unused arrays.

    S = load(filename, variableName);
    x = S.(variableName);
end

function [phaseImage, fieldPpm] = preparePhaseData(fieldVol, mask, opts, phaseScale)
    if ~isreal(fieldVol)
        error('qsm_lamdi:InvalidPhaseInput', ...
            ['The phase/local-field input must be real-valued. If the MAT file ' ...
             'contains complex GRE data, pass angle(data) or a preprocessed local phase/field map.']);
    end

    echoTimes = opts.echoTimes(:).';
    nEcho = numel(echoTimes);
    nVox = nnz(mask);
    fieldStack = makeEchoStack(fieldVol, nEcho, 'local field/phase');
    phaseImage = zeros(nVox, nEcho);
    fieldPpmByEcho = zeros(nVox, nEcho);

    for echoIdx = 1:nEcho
        srcIdx = min(echoIdx, size(fieldStack, 4));
        vol = fieldStack(:, :, :, srcIdx);
        vec = double(vol(mask==1));

        switch lower(opts.fieldMapUnits)
            case {'ppm'}
                phaseImage(:, echoIdx) = vec * phaseScale(echoIdx);
                fieldPpmByEcho(:, echoIdx) = vec;
            case {'radian'}
                phaseImage(:, echoIdx) = vec;
                fieldPpmByEcho(:, echoIdx) = vec / phaseScale(echoIdx);
            otherwise
                error('qsm_lamdi:InvalidUnits', ...
                    'opts.fieldMapUnits must be ''ppm'', or ''radian''.');
        end
    end
    
    % average the local field map in ppm
    fieldPpm = mean(fieldPpmByEcho, 2);
end

function stack = makeEchoStack(vol, nEcho, label)
    if ndims(vol) == 3
        stack = reshape(vol, [size(vol), 1]);
    elseif ndims(vol) == 4
        if size(vol, 4) ~= nEcho && size(vol, 4) ~= 1
            error('qsm_lamdi:EchoMismatch', ...
                'The 4-D %s array has %d echoes, but opts.echoTimes has %d entries.', ...
                label, size(vol, 4), nEcho);
        end
        stack = vol;
    else
        error('qsm_lamdi:InvalidArray', 'The %s array must be 3-D or 4-D.', label);
    end
end

function [weights, iMag] = prepareMagnitudeWeights(magnitude, mask, nEcho, opts)
    nVox = nnz(mask);
    if isempty(magnitude)
        weights = ones(nVox, nEcho);
        iMag = double(mask);
        return;
    end

    maskSize = size(mask);
    if size(magnitude, 1) ~= maskSize(1) || size(magnitude, 2) ~= maskSize(2) || ...
            size(magnitude, 3) ~= maskSize(3)
        error('qsm_lamdi:MagnitudeSizeMismatch', ...
            'Magnitude data and mask must have matching spatial dimensions.');
    end

    magnitude = abs(double(magnitude));
    if ndims(magnitude) == 3
        iMag = magnitude;
        weights = repmat(iMag(mask==1), 1, nEcho);
    elseif ndims(magnitude) == 4
        nMagEcho = size(magnitude, 4);
        magByEcho = zeros(nVox, nMagEcho);
        for echoIdx = 1:nMagEcho
            tmp = magnitude(:, :, :, echoIdx);
            magByEcho(:, echoIdx) = tmp(mask==1);
        end
        iMag = sqrt(sum(magnitude.^2, 4));
        rssWeight = sqrt(sum(magByEcho.^2, 2));

        if opts.useEchoSpecificMagnitudeWeights
            if nMagEcho ~= nEcho
                error('qsm_lamdi:MagnitudeEchoMismatch', ...
                    ['opts.useEchoSpecificMagnitudeWeights is true, but the magnitude data ' ...
                     'have %d echoes while opts.echoTimes has %d entries.'], nMagEcho, nEcho);
            end
            weights = magByEcho;
        else
            weights = repmat(rssWeight, 1, nEcho);
        end
    else
        error('qsm_lamdi:InvalidMagnitude', 'Magnitude data must be 3-D or 4-D.');
    end

    weights(~isfinite(weights)) = 0;
end

function inputPar = initializeInputParameters(initCoeff, opts)
    baseLambda = 1 / max(mean(abs(initCoeff)), realmin);

    inputPar.lambda_1 = baseLambda;
    inputPar.lambda_2 = opts.initialSecondLambdaScale * baseLambda;
    inputPar.lap_w1 = opts.initialMixtureWeights(1);
    inputPar.lap_w2 = opts.initialMixtureWeights(2);
end

function phaseEstimate = applyPhaseOperator(A, chi, phaseScale, nEcho)
    fieldEstimate = real(A.times(chi));
    fieldEstimate = fieldEstimate(:);
    phaseEstimate = zeros(numel(fieldEstimate), nEcho);
    for echoIdx = 1:nEcho
        phaseEstimate(:, echoIdx) = fieldEstimate * phaseScale(echoIdx);
    end
end

function residueWeight = updateResidualWeights(residual, residualSigma)
    residueWeight = ones(size(residual));
    if isempty(residualSigma) || isinf(residualSigma) || residualSigma <= 0
        return;
    end

    residualStd = sqrt(var(real(residual(:))) + var(imag(residual(:))));
    if residualStd <= 0 || ~isfinite(residualStd)
        return;
    end

    bad = abs(residual) >= residualSigma * residualStd;
    residueWeight(bad) = 1 ./ ((abs(residual(bad)) / residualStd).^2);
end

function coeff = extractWaveletCoefficients(waveletStruct)
    %EXTRACTWAVELETCOEFFICIENTS Convert a wavedec3 coefficient structure to a vector.
    %
    % MATLAB's wavedec3 stores the coefficient blocks in the cell array
    % waveletStruct.dec.  The AMP solver works with a single vector, so we
    % concatenate the coefficient blocks in the exact order used by the template.

    if ~isstruct(waveletStruct) || ~isfield(waveletStruct, 'dec')
        error('qsm_lamdi:InvalidWaveletStruct', ...
            'The input must be a wavedec3-style structure with a dec field.');
    end

    if iscell(waveletStruct.dec)
        nCoeff = 0;
        for idx = 1:numel(waveletStruct.dec)
            nCoeff = nCoeff + numel(waveletStruct.dec{idx});
        end

        coeff = zeros(nCoeff, 1, 'like', waveletStruct.dec{1});
        offset = 0;
        for idx = 1:numel(waveletStruct.dec)
            block = waveletStruct.dec{idx};
            nBlock = numel(block);
            coeff(offset + (1:nBlock)) = block(:);
            offset = offset + nBlock;
        end
    elseif isnumeric(waveletStruct.dec)
        % Fallback for nonstandard templates in which dec is stored as an array.
        coeff = waveletStruct.dec(:);
    else
        error('qsm_lamdi:InvalidWaveletStruct', ...
            'waveletStruct.dec must be either a cell array or a numeric array.');
    end
end

function waveletStruct = buildWaveletStruct(coeff, template)
    %BUILDWAVELETSTRUCT Rebuild a wavedec3 coefficient structure from a vector.
    %
    % The block sizes and ordering are taken from template.dec, so this function
    % is the inverse of extractWaveletCoefficients when called with the same
    % template structure.

    if ~isstruct(template) || ~isfield(template, 'dec')
        error('qsm_lamdi:InvalidWaveletTemplate', ...
            'The template must be a wavedec3-style structure with a dec field.');
    end

    coeff = coeff(:);
    waveletStruct = template;

    if iscell(template.dec)
        offset = 0;
        for idx = 1:numel(template.dec)
            nBlock = numel(template.dec{idx});
            nextOffset = offset + nBlock;
            if nextOffset > numel(coeff)
                error('qsm_lamdi:CoefficientLengthMismatch', ...
                    'The coefficient vector is shorter than required by the wavelet template.');
            end
            waveletStruct.dec{idx} = reshape(coeff(offset + (1:nBlock)), size(template.dec{idx}));
            offset = nextOffset;
        end

        if offset ~= numel(coeff)
            error('qsm_lamdi:CoefficientLengthMismatch', ...
                'The coefficient vector length does not match the wavelet template.');
        end
    elseif isnumeric(template.dec)
        if numel(coeff) ~= numel(template.dec)
            error('qsm_lamdi:CoefficientLengthMismatch', ...
                'The coefficient vector length does not match the wavelet template.');
        end
        waveletStruct.dec = reshape(coeff, size(template.dec));
    else
        error('qsm_lamdi:InvalidWaveletTemplate', ...
            'template.dec must be either a cell array or a numeric array.');
    end
end


