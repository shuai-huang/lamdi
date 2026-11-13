# LAMDI MATLAB Code

This folder contains a cleaned MATLAB implementation of **[Laplace-Mixture Dipole Inversion (LAMDI) for Quantitative Susceptibility Mapping](https://arxiv.org/pdf/2606.10240)** 

* If you use this code and find it helpful, please cite the above paper. Thanks :smile:
```
@article{lamdi_QSM_2026,
  title={Laplace-Mixture Dipole Inversion for Quantitative Susceptibility Mapping},
  author={Huang, Shuai and Lah, James J. and Allen, Jason W. and Qiu, Deqiang},
  journal={arXiv preprint arXiv:2606.10240},
  year={2026}
}
```

## Files

- `qsm_lamdi.m` — main reconstruction entry point. This version reads inputs from MATLAB arrays or a MATLAB `.mat` file only.
- `gamp_lap_mix.m` — damped AMP-PE updates with a two-component Laplace mixture prior.
- `A_lamdi_LinTrans.m` — linearized QSM measurement operator in the wavelet domain.
- `A_wavelet_3d_LinTrans.m` — 3-D wavelet transform operator wrapped as a `LinTrans` object.

## External dependencies

The code expects your existing `functions` folder to be on the MATLAB path. In particular, the main routine calls the following utilities:

- `LinTrans`
- `GenerateDipoleFT3Drot`
- `SI_operator_withmask`
- `dipole_kernel_angulated`
- `chiL2`

The MATLAB Wavelet Toolbox is also required for `wavedec3`, `waverec3`, and `dwtmode`.

## Expected input format

The simplest input is a single MATLAB file containing at least:

- `phase` — a 3-D or 4-D local phase/field array, please make sure you change the `fieldMapUnits` option to `ppm` or `radian` accordingly.
- `mask` — a 3-D brain mask.
- `magnitude` — optional 3-D or 4-D magnitude image (for multi-echo data). If omitted, unit weights are used.

If your variables have different names, set `opts.phaseVariable`, `opts.maskVariable`, and `opts.magnitudeVariable` explicitly.


## Basic usage

```matlab
% The first example is single-echo reconstruction, the second example is multi-echo reconstruction
% Multi-echo reconstruction seems to produce QSM with better quality and more details

% 1) Single-echo reconstruction

opts = struct();
opts.functionsDir = './functions';

% MAT file containing phase/field, mask, and optional magnitude data.
opts.dataFile = './data/test_data.mat';
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
opts.waveletLevel = 4;


% If AMP does not converge, the damping rate should be reduced and the number of iterations should be increased accordingly
opts.gampDampingAlpha = 0.005;
opts.gampDampingBeta = 0.1;
opts.gampMaxIterations = 5;
opts.gampParameterIterations = 5;

% The number of linearizations of the nonlinear measurement model
% and residual sigma criterion used to filter out corrupted voxels
opts.maxOuterIterations = 100;
opts.useResidueWeight = true;
opts.residualSigma = 10;

[chi, result] = qsm_lamdi(opts);
save('chi_and_result_single_echo.mat', 'chi', 'result', '-v7.3')

% 2) Multi-echo reconstruction

opts = struct();
opts.functionsDir = './functions';

% MAT file containing phase/field, mask, and optional magnitude data.
opts.dataFile = strcat('./data/test_data_multi_echo.mat');
opts.phaseVariable = 'phase';         % 3-D/4-D phase or local field
opts.maskVariable = 'mask';           % 3-D brain mask
opts.magnitudeVariable = 'magnitude'; % optional; use 'data' for complex GRE data if needed

% Please make sure to match these to your input data.
% Use 'ppm' if the input is local field in ppm, and 'radian' if it is the actual phase.
% The unit of echo time is 'second'. There can be multiple echo times.
% The unit of field strength is 'Tesla'.
% The unit of voxel size is 'mm'.
% The rotation Matrix is a 3D matrix.

opts.fieldMapUnits = 'radian';
opts.echoTimes = [7.7 13.4 18.8 25.3 31.7 38.2 ]*1e-3;
opts.fieldStrengthT = 3;
opts.voxelSize = [1 1 1];
opts.rotationMatrix = eye(3);

% The wavelet transform
opts.waveletName = 'db1';
opts.waveletLevel = 4;

% If AMP does not converge, the damping rate should be reduced and the number of iterations should be increased accordingly
opts.gampDampingAlpha = 0.005;
opts.gampDampingBeta = 0.1;
opts.gampMaxIterations = 5;
opts.gampParameterIterations = 5;

% The number of linearizations of the nonlinear measurement model
% and residual sigma criterion used to filter out corrupted voxels
opts.maxOuterIterations = 100;
opts.useResidueWeight = true;
opts.residualSigma = 10;

[chi, result] = qsm_lamdi(opts);
save('chi_and_result_multi_echo.mat', 'chi', 'result', '-v7.3')

```

## Notes

The input phase/field must be real-valued and can be 3-D or 4-D. If it is 3-D and multiple echo times are supplied, the same input volume is used for each echo. The forward model internally uses `opts.echoTimes`, `opts.fieldStrengthT`, and `opts.gyromagnetic_ratio` to convert susceptibility in ppm to phase in radians if needed. The output susceptibility map `chi` is in ppm.
