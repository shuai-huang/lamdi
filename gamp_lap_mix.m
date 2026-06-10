function [res, input_par, output_par] = gamp_lap_mix(A_lamdi, y, gamp_par, input_par, output_par)
%GAMP_LAP_MIX GAMP updates with a two-component Laplace mixture prior.
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
%       tau_w                complex Gaussian noise variance
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

tau_w = max(output_par.tau_w, realmin);

convergence = nan(max_pe_spar_ite, 1);
for ite_pe = 1:max_pe_spar_ite
    tau_p_meas = A_lamdi.multSq(tau_x_meas);
    p_hat_meas = A_lamdi.mult(x_hat_meas) - tau_p_meas .* s_hat_meas;

    for ite_pe_est = 1:max_pe_est_ite %#ok<NASGU>
        tau_w = estimateOutputVariance(y, tau_w, p_hat_meas, tau_p_meas, damping_rate_beta);
    end

    tau_s_meas = 1 ./ max(tau_w + tau_p_meas, realmin);
    s_hat_meas = (y - p_hat_meas) .* tau_s_meas;

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
                 'mix=(%0.3f,%0.3f), tau_w=%0.3e\n'], ...
            ite_pe, convergence(ite_pe), lambda_1, lambda_2, lap_w1, lap_w2, tau_w);
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

output_par.tau_w = tau_w;
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

function tau_w = estimateOutputVariance(y, tau_w, p_hat, tau_p, damping_rate_beta)
tauNew = mean(abs(p_hat(:) - y(:)).^2 + real(tau_p(:)));
tauNew = max(real(tauNew), realmin);
tau_w = tau_w + damping_rate_beta * (tauNew - tau_w);
tau_w = max(real(tau_w), realmin);
end

