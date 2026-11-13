function [res, input_par, output_par] = gamp_lap_mix_gm(A_lamdi, y, gamp_par, input_par, output_par)
% GAMP_LAP_MIX_GM GAMP updates with a two-component Laplace mixture signal prior and two-component Gaussian mixture noise prior.
% GAMP_LAP_MIX contains a singla component Gaussian noise prior, it must be run first to produce an initial solution
% The implementation of the two-component Gaussian noise prior has redundancy built into it.
% The "second" Gaussian component is used to model the outliers, it has zero mean
% The "first" Gaussian component is used to model the majority of the noise, but it is implemented as a Gaussian mixture with redundancy for further analysis. In practice, you can simply set the number of Gaussian mixtures to 1
%
%   [RES, INPUT_PAR, OUTPUT_PAR] = GAMP_LAP_MIX(A, Y, GAMP_PAR,
%   INPUT_PAR, OUTPUT_PAR) performs damped AMP-PE iterations for the
%   linearized LAMDI measurement model.
%
%   INPUT_PAR fields:
%       lambda_1, lambda_2   Laplace rate parameters
%       lap_w1, lap_w2       mixture weights, summing to one
%
%   OUTPUT_PAR fields:
%       theta, phi, omega, num_c    The "first" Gaussian component: mean, variance, inner mixture weight, the number of Gaussian components (set to 1)
%       psi, gamma              The second zero-mean Gaussian component: variance, mixture weight
%
%   GAMP_PAR fields:
%       x_hat_meas, tau_x_meas, s_hat_meas
%       max_pe_spar_ite, max_pe_est_ite
%       damping_rate_alpha, damping_rate_beta, cvg_thd
%       verbose              optional logical flag

max_pe_spar_ite = gamp_par.max_pe_spar_ite;
max_pe_est_ite = gamp_par.max_pe_est_ite;
cvg_thd = gamp_par.cvg_thd;
damping_rate_beta = gamp_par.damping_rate_beta;
damping_rate_alpha = gamp_par.damping_rate_alpha;
verbose = getStructField(gamp_par, 'verbose', false);

x_hat_meas = gamp_par.x_hat_meas;
tau_x_meas = gamp_par.tau_x_meas;
s_hat_meas = gamp_par.s_hat_meas;

lambda_1 = max(input_par.lambda_1, realmin);
lambda_2 = max(input_par.lambda_2, realmin);
lap_w1 = input_par.lap_w1;
lap_w2 = input_par.lap_w2;
[lap_w1, lap_w2] = normalizeTwoWeights(lap_w1, lap_w2);

theta = output_par.theta;
phi = output_par.phi;
omega = output_par.omega;
num_c = output_par.num_c;

psi = output_par.psi;
gamma = output_par.gamma;

convergence = nan(max_pe_spar_ite, 1);
for ite_pe = 1:max_pe_spar_ite
    tau_p_meas = A_lamdi.multSq(tau_x_meas);
    p_hat_meas = A_lamdi.mult(x_hat_meas) - tau_p_meas .* s_hat_meas;

    for ite_pe_est = 1:max_pe_est_ite %#ok<NASGU>
        [theta, phi, omega, psi, gamma] = estimateOutputVariance(y(:)-p_hat_meas(:), tau_p_meas(:), theta, phi, omega, psi, gamma, damping_rate_beta);
    end

    [noise_update, tau_noise_update] = output_function(y(:)-p_hat_meas(:), tau_p_meas(:), theta, phi, omega, psi, gamma);
    
    z_hat = y-reshape(noise_update,size(y));
    tau_z = reshape(tau_noise_update, size(tau_p_meas));

    tau_s_meas = 1./tau_p_meas.*(1-tau_z./tau_p_meas);
    s_hat_meas = 1./tau_p_meas.*(z_hat-p_hat_meas);

    tau_r_meas = 1 ./ max(A_lamdi.multSqTr(tau_s_meas), realmin);
    r_hat_meas = x_hat_meas + tau_r_meas .* A_lamdi.multTr(s_hat_meas);

    % Susceptibility and wavelet coefficients are real-valued.  Small
    % imaginary components can arise numerically because the measurement
    % model is complex-valued.
    
    %if verbose && norm(imag(r_hat_meas(:)), 'fro') > 1e-8 * max(norm(real(r_hat_meas(:)), 'fro'), eps)
    %    fprintf('gamp_lap_mix: discarding small imaginary component in r_hat (relative norm %0.3e).\n', ...
    %        norm(imag(r_hat_meas(:)), 'fro') / max(norm(real(r_hat_meas(:)), 'fro'), eps));
    %end
    r_hat_meas = real(r_hat_meas);

    resp1 = computeResponsibilities(r_hat_meas, lambda_1, lambda_2, lap_w1, lap_w2);
    resp2 = 1 - resp1;

    for ite_pe_est = 1:max_pe_est_ite %#ok<NASGU>
        lambda_1 = estimateInputLambda(r_hat_meas, lambda_1, resp1, damping_rate_beta);
        lambda_2 = estimateInputLambda(r_hat_meas, lambda_2, resp2, damping_rate_beta);
    end

    lap_w1 = sum(resp1(:)) / max(numel(resp1), 1);
    lap_w2 = 1 - lap_w1;
    [lap_w1, lap_w2] = normalizeTwoWeights(lap_w1, lap_w2);

    % Keep the labels interpretable: component 1 is the narrow/small-gradient
    % component and therefore has the larger Laplace rate.
    if lambda_1 < lambda_2
        [lambda_1, lambda_2] = deal(lambda_2, lambda_1);
        [lap_w1, lap_w2] = deal(lap_w2, lap_w1);
    end

    shrinkWeight = computeShrinkageWeight(r_hat_meas, lambda_1, lambda_2, lap_w1, lap_w2);

    x_hat_prev = x_hat_meas;
    tau_x_prev = tau_x_meas;

    [x_hat_undamped, tau_x_undamped] = inputDenoiser(r_hat_meas, tau_r_meas, shrinkWeight);

    x_hat_meas = x_hat_prev + damping_rate_alpha * (x_hat_undamped - x_hat_prev);
    tau_x_meas = tau_x_prev + damping_rate_alpha * (tau_x_undamped - tau_x_prev);
    tau_x_meas = max(real(tau_x_meas), 0);

    convergence(ite_pe) = norm(x_hat_meas(:) - x_hat_prev(:), 'fro') / ...
        max(norm(x_hat_meas(:), 'fro'), eps);

    if verbose
        fprintf(['GAMP iter %03d: cvg=%0.3e, lambda=(%0.3e,%0.3e), ' ...
                 'mix=(%0.3f,%0.3f), phi=%0.3e, psi=%0.3e, gamma=%0.3e\n'], ...
            ite_pe, convergence(ite_pe), lambda_1, lambda_2, lap_w1, lap_w2, phi, psi, gamma);
    end

    if convergence(ite_pe) < cvg_thd
        convergence = convergence(1:ite_pe);
        break;
    end
end

res = struct();
res.x_hat_meas = x_hat_meas;
res.tau_x_meas = tau_x_meas;
res.s_hat_meas = s_hat_meas;
res.convergence = convergence;

input_par.lambda_1 = lambda_1;
input_par.lambda_2 = lambda_2;
input_par.lap_w1 = lap_w1;
input_par.lap_w2 = lap_w2;

output_par.theta = theta;
output_par.phi = phi;
output_par.omega = omega;
output_par.num_c = num_c;

output_par.psi = psi;
output_par.gamma = gamma;

end

function value = getStructField(s, fieldName, defaultValue)
if isfield(s, fieldName)
    value = s.(fieldName);
else
    value = defaultValue;
end
end

function [w1, w2] = normalizeTwoWeights(w1, w2)
w1 = max(real(w1), 0);
w2 = max(real(w2), 0);
normalizer = w1 + w2;
if normalizer <= 0 || ~isfinite(normalizer)
    w1 = 0.5;
    w2 = 0.5;
else
    w1 = w1 / normalizer;
    w2 = w2 / normalizer;
end
end

function resp1 = computeResponsibilities(r_hat, lambda_1, lambda_2, lap_w1, lap_w2)
logp1 = log(max(lap_w1, realmin)) + log(max(lambda_1, realmin)) - lambda_1 * abs(r_hat);
logp2 = log(max(lap_w2, realmin)) + log(max(lambda_2, realmin)) - lambda_2 * abs(r_hat);
maxLog = max(logp1, logp2);
p1 = exp(logp1 - maxLog);
p2 = exp(logp2 - maxLog);
resp1 = p1 ./ max(p1 + p2, realmin);
end

function shrinkWeight = computeShrinkageWeight(r_hat, lambda_1, lambda_2, lap_w1, lap_w2)
logp1 = log(max(lap_w1, realmin)) + log(max(lambda_1, realmin)) - lambda_1 * abs(r_hat);
logp2 = log(max(lap_w2, realmin)) + log(max(lambda_2, realmin)) - lambda_2 * abs(r_hat);
maxLog = max(logp1, logp2);
p1 = exp(logp1 - maxLog);
p2 = exp(logp2 - maxLog);
shrinkWeight = (p1 * lambda_1 + p2 * lambda_2) ./ max(p1 + p2, realmin);
end

function [x_hat, tau_x] = inputDenoiser(r_hat, tau_r, shrinkWeight)
threshold = shrinkWeight .* tau_r;
x_hat = max(0, abs(r_hat) - threshold) .* sign(r_hat);

tau_x = tau_r;
tau_x(abs(x_hat) == 0) = 0;
tau_x = max(real(tau_x), 0);
end

function lambda = estimateInputLambda(r_hat, lambda, responsibility, damping_rate_beta)
weightedMeanAbs = sum(responsibility(:) .* abs(r_hat(:))) / max(sum(responsibility(:)), realmin);
lambdaNew = 1 / max(weightedMeanAbs, realmin);
lambdaNew = max(lambdaNew, realmin);
lambda = lambda + damping_rate_beta * (lambdaNew - lambda);
lambda = max(real(lambda), realmin);
end

function [theta, phi, omega, psi, gamma] = estimateOutputVariance(h_hat, tau_h, theta, phi, omega, psi, gamma, damping_rate_beta)

    omega_pre = omega;
    theta_pre = theta;
    phi_pre = phi;
    gamma_pre = gamma;
    psi_pre = psi;

    % do we need some kind of normalization here?
    dim_smp=length(h_hat);
    num_cluster=length(omega);

    rho_tmp_mat_1 = zeros(dim_smp, num_cluster);
    for (i = 1:num_cluster)
        rho_tmp_mat_1(:,i) = (1-gamma) * omega(i) * 1./(tau_h+phi(i)) .* exp( - (abs(h_hat-theta(i)) ./ sqrt(tau_h+phi(i))).^2 );
    end

    rho_tmp_2 = gamma * 1./(tau_h+psi) .* exp( -(abs(h_hat) ./ sqrt(tau_h+psi)).^2);
    rho_tmp_1 = sum(rho_tmp_mat_1, 2);

    rho_tmp_sum = rho_tmp_1 + rho_tmp_2;
    rho_tmp_sum_zero_idx = (rho_tmp_sum==0);    % leave the outliers to the first Gaussian component


    rho_block_1 = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        rho_block_1(:,i) = rho_tmp_mat_1(:,i) ./ rho_tmp_sum;
        rho_block_1(rho_tmp_sum_zero_idx,i) = 0;
    end

    rho_block_2 = rho_tmp_2 ./ rho_tmp_sum;
    rho_block_2(rho_tmp_sum_zero_idx) = 1;


    % compute the mixture weight of the gaussian distributions: omega
    omega_new = sum(rho_block_1);
    omega_new = omega_new.';    % this is transpose not conjungate transpose
    omega_new = omega_new / sum(omega_new);

    omega = omega + damping_rate_beta * (omega_new - omega);

    % compute the weight of the outlier distribution: gamma
    %gamma_new = sum(rho_block_2) / (sum(sum(rho_block_1)) + sum(rho_block_2));
    %gamma = gamma + damping_rate_beta * (gamma_new - gamma);

    % compute the means of the gaussian distributions: theta
    theta_tmp_mat = rho_block_1;
    theta_tmp_mat_sum_1 = sum(theta_tmp_mat);
    zero_idx=(theta_tmp_mat_sum_1==0);  % find the guassian mixtures with zero weight
    % do not remove the following code, reserved for future analysis
    if (0)
        theta_tmp_mat_1 = theta_tmp_mat;
        theta_tmp_mat_2 = theta_tmp_mat;
        for (i=1:num_cluster)
            theta_tmp_mat_1(:,i) = theta_tmp_mat_1(:,i) .* ( h_hat ./ (phi(i)+tau_h) );
            theta_tmp_mat_2(:,i) = theta_tmp_mat_2(:,i) ./ (phi(i)+tau_h);
        end
        theta_new = sum(theta_tmp_mat_1) ./ sum(theta_tmp_mat_2);   % to avoid division by 0
        theta_new(zero_idx) = theta(zero_idx);  % keep the gaussian mixtures with zero weight fixed

        theta_new_inf_idx = isinf(theta_new);
        theta_new(theta_new_inf_idx) = theta(theta_new_inf_idx); % keep the gaussian mixtures with infinity theta fixed just in case

        theta_new = theta_new.';    % this is transpose, not conjungate transpose

        theta = theta + damping_rate_beta * (theta_new - theta);
    end
    % set the means of the gaussian distributions to zeros
    theta = zeros(size(theta));

    % compute the variances of the gaussian distributions: phi
    % what happens if theta_tmp_mat is 0/0
    phi_tmp_mat_1 = rho_block_1;
    phi_tmp_mat_2 = rho_block_1;
    for (i=1:num_cluster)
        phi_tmp_mat_1(:,i) = phi_tmp_mat_1(:,i) .* ((abs(h_hat-theta(i))).^2 + tau_h);
    end
    phi_new = sum(phi_tmp_mat_1) ./ sum(phi_tmp_mat_2);   % since the coefficients in tau_h are the same, just use tau_h(1)
    phi_new(zero_idx) = phi(zero_idx);  % keep the gaussian mixtures with zero weight fixed

    phi_new_inf_idx = isinf(phi_new);
    phi_new(phi_new_inf_idx) = phi(phi_new_inf_idx); % keep the gaussian mixtures with infinity phi fixed just in case

    phi_new = phi_new';
    % double check
    for (i=1:num_cluster)
        if (phi_new(i)<0)
            phi_new(i) = phi(i);
        end
    end

    phi = phi + damping_rate_beta * (phi_new - phi);

    % compute the variance of the outlier distribution: psi
    psi_tmp_mat = rho_block_2;
    psi_tmp_mat_1 = psi_tmp_mat;
    psi_tmp_mat_2 = psi_tmp_mat;
    psi_new = sum(psi_tmp_mat_1 .* ((abs(h_hat)).^2+tau_h)) / sum(psi_tmp_mat_2);    % since the coefficients in tau_h are the same, just use tau_h(1)

    if (psi_new<0)
        psi_new = psi;
    end

    psi = psi + damping_rate_beta * (psi_new - psi);

end


function [noise_update, tau_noise_update] = output_function(h_hat, tau_h, theta, phi, omega, psi, gamma)

    dim_smp=length(h_hat);
    num_cluster=length(omega);

    block_mat = zeros(dim_smp, num_cluster);
    for (i=1:num_cluster)
        block_mat(:,i) = (1-gamma) * omega(i) * (tau_h./(phi(i)+tau_h)) .* exp( -(abs(theta(i)-h_hat) ./ sqrt(phi(i)+tau_h)).^2 );
    end

    % compute noise_update
    block_mat_nmr_x = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x(:,i) = block_mat_nmr_x(:,i) .* (theta(i) * tau_h + h_hat * phi(i)) ./ (phi(i) + tau_h);
    end

    block_mat_2 = gamma * (tau_h./(psi+tau_h)) .* exp( -(abs(h_hat) ./ sqrt(psi+tau_h)).^2 );
    block_mat_nrm_x_2 = block_mat_2 .* (h_hat*psi ./ (psi+tau_h));

    nmr_x = sum(block_mat_nmr_x, 2) + block_mat_nrm_x_2;
    dnm_x = sum(block_mat, 2) + block_mat_2;

    noise_update = nmr_x ./ (dnm_x);

    % if dnm_x is zero, set noise_update to h_hat
    dnm_x_zero_idx = (dnm_x==0);
    noise_update(dnm_x_zero_idx) = h_hat(dnm_x_zero_idx);

    % compute tau_noise_update
    block_mat_nmr_x_sq = block_mat;
    for (i=1:num_cluster)
        block_mat_nmr_x_sq(:,i) = block_mat_nmr_x_sq(:,i) .* (  phi(i)*tau_h./(phi(i) + tau_h) + (abs( (theta(i)*tau_h + h_hat*phi(i)) ./ (phi(i) + tau_h) )).^2  );
    end

    block_mat_nmr_x_sq_2 = block_mat_2 .* ( psi*tau_h./(psi+tau_h) + (abs(h_hat*psi) ./ (psi+tau_h)).^2 );

    nmr_x_sq = sum(block_mat_nmr_x_sq, 2) + block_mat_nmr_x_sq_2;
    dnm_x_sq = dnm_x;

    tau_noise_update_seq = (nmr_x_sq ./ dnm_x_sq - (abs(noise_update)).^2);  % this is nonnegative in theory

    % if dnm_x is zero, set tau_noise_update_seq to 0
    tau_noise_update_seq(dnm_x_zero_idx) = 0;

    tau_noise_update = repmat(mean(tau_noise_update_seq),size(tau_noise_update_seq));
    tau_noise_update = max(tau_noise_update, 1e-12);  % just in case

end

