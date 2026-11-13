function foo = qsm_rec(subject_idx) 

opts = struct();
opts.functionsDir = './functions';

% MAT file containing phase/field, mask, and optional magnitude data.
opts.dataFile = strcat('./data/test_data_',subject_idx,'.mat');
opts.phaseVariable = 'phase';         % 3-D/4-D phase or local field
opts.maskVariable = 'mask';           % 3-D brain mask
opts.magnitudeVariable = 'magnitude'; % optional; use 'data' for complex GRE data if needed

% Please make sure to match these to your input data.
% Use 'ppm' if the input is local field in ppm, and 'radian' if it is the actual phase.
% The unit of echo time is 'second'. There can be multiple echo times.
% The unit of field strength is 'Tesla'.
% The unit of voxel size is 'mm'.
% The rotation Matrix is a 3D matrix.

opts.fieldMapUnits = 'ppm';
opts.echoTimes = 8e-3;
opts.fieldStrengthT = 3;
opts.voxelSize = [1 1 1];
opts.rotationMatrix = eye(3);

% The wavelet transform
opts.waveletName = 'db1';
opts.waveletLevel = 3;


% If AMP does not converge, the damping rate should be reduced and the number of iterations should be increased accordingly
opts.gampDampingAlpha = 0.01;
opts.gampDampingBeta = 0.1;
opts.gampMaxIterations = 5;
opts.gampParameterIterations = 5;

% The number of linearizations of the nonlinear measurement model
% and residual sigma criterion used to filter out corrupted voxels
opts.maxOuterIterations = 100;
opts.useResidueWeight = true;
opts.residualSigma = 10;

[chi, result] = qsm_lamdi(opts);
save(strcat('./results/chi_residue_weight_',subject_idx,'.mat'), 'chi', '-v7.3')

end
