classdef A_lamdi_LinTrans < LinTrans
    %A_LAMDI_LINTRANS Linearized LAMDI measurement operator.
    %
    %   This class represents the wavelet-domain linear operator used by
    %   LAMDI inside each linearized nonlinear QSM update:
    %
    %       y_e = W_e .* A( H^{-1} v ) * phaseScale(e),
    %
    %   where v contains wavelet coefficients, H^{-1} is implemented by the
    %   inverse wavelet transform, A is the masked dipole operator, and W_e
    %   contains the complex first-order Taylor derivative together with the
    %   magnitude and residual weights.

    properties
        M               % Total output dimension: nVoxels * nEchoes
        N               % Input dimension: number of wavelet coefficients
        mat_sz          % Reconstructed volume size
        W               % Measurement weights, nVoxels-by-nEchoes
        Wt              % Conjugate measurement weights
        A               % Masked dipole operator with times/trans methods
        Psit            % Inverse wavelet transform: struct -> volume
        Psi             % Forward wavelet transform: volume -> struct
        E               % Coefficients -> wavelet struct
        Et              % Wavelet struct -> coefficients
        phaseScale      % ppm-to-radian phase scale for each echo
        M_row           % Number of voxels inside the mask
        M_col           % Number of echo/time points
        S               % Optional exact squared forward operator
        St              % Optional exact squared adjoint operator
    end

    methods
        function obj = A_lamdi_LinTrans(M, N, mat_sz, W, A, phaseScale, ...
                Psit, Psi, E, Et, M_row, M_col, S, St)
            obj = obj@LinTrans;

            validateattributes(M, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(N, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(mat_sz, {'numeric'}, {'vector', 'numel', 3});

            obj.M = double(M);
            obj.N = double(N);
            obj.mat_sz = double(mat_sz(:).');
            obj.M_row = double(M_row);
            obj.M_col = double(M_col);

            if obj.M ~= obj.M_row * obj.M_col
                error('A_lamdi_LinTrans:DimensionMismatch', ...
                    'M must equal M_row * M_col.');
            end

            if size(W, 1) ~= obj.M_row || size(W, 2) ~= obj.M_col
                error('A_lamdi_LinTrans:WeightSizeMismatch', ...
                    'W must have size M_row-by-M_col.');
            end

            if isscalar(phaseScale)
                phaseScale = repmat(phaseScale, 1, obj.M_col);
            end
            if numel(phaseScale) ~= obj.M_col
                error('A_lamdi_LinTrans:PhaseScaleMismatch', ...
                    'phaseScale must be scalar or have one value per echo.');
            end

            if ~isa(Psit, 'function_handle') || ~isa(Psi, 'function_handle') || ...
                    ~isa(E, 'function_handle') || ~isa(Et, 'function_handle')
                error('A_lamdi_LinTrans:InvalidFunctionHandle', ...
                    'Psit, Psi, E, and Et must be function handles.');
            end

            obj.W = W;
            obj.Wt = conj(W);
            obj.A = A;
            obj.phaseScale = phaseScale(:).';
            obj.Psit = Psit;
            obj.Psi = Psi;
            obj.E = E;
            obj.Et = Et;

            if nargin >= 13 && ~isempty(S)
                if isnumeric(S) && isscalar(S) && S > 0
                    obj.FrobNorm = S;
                elseif nargin >= 14 && isa(S, 'function_handle') && isa(St, 'function_handle')
                    obj.S = S;
                    obj.St = St;
                    obj.FrobNorm = [];
                else
                    error('A_lamdi_LinTrans:InvalidOptionalInputs', ...
                        'Optional inputs must be either FrobNorm or S/St function handles.');
                end
            else
                obj.FrobNorm = obj.estimateFrobeniusNorm(2);
            end
        end

        function [m, n] = size(obj, dim)
            if nargin > 1
                switch dim
                    case 1
                        m = obj.M;
                    case 2
                        m = obj.N;
                    otherwise
                        m = 1;
                end
            elseif nargout < 2
                m = [obj.M, obj.N];
            else
                m = obj.M;
                n = obj.N;
            end
        end

        function y = mult(obj, x)
            if numel(x) ~= obj.N
                error('A_lamdi_LinTrans:InputSizeMismatch', ...
                    'Input vector length must equal N.');
            end

            chi = obj.Psit(obj.E(x(:)));
            dipoleResponse = obj.A.times(chi);
            dipoleResponse = dipoleResponse(:);

            y = zeros(obj.M_row, obj.M_col);
            for echoIdx = 1:obj.M_col
                y(:, echoIdx) = dipoleResponse .* obj.W(:, echoIdx) .* obj.phaseScale(echoIdx);
            end
        end

        function x = multTr(obj, y)
            if numel(y) ~= obj.M
                error('A_lamdi_LinTrans:OutputSizeMismatch', ...
                    'Input to multTr must have M elements.');
            end
            y = reshape(y, obj.M_row, obj.M_col);

            weightedDataSum = zeros(obj.M_row, 1);
            for echoIdx = 1:obj.M_col
                weightedDataSum = weightedDataSum + ...
                    y(:, echoIdx) .* obj.Wt(:, echoIdx) .* obj.phaseScale(echoIdx);
            end

            imageData = reshape(obj.A.trans(weightedDataSum), obj.mat_sz);
            x = obj.Et(obj.Psi(imageData));
        end

        function y = multSq(obj, x)
            if isempty(obj.FrobNorm)
                y = obj.S(x);
            else
                scale = (obj.FrobNorm^2 / (obj.M * obj.N)) * sum(x(:));
                y = ones(obj.M_row, obj.M_col) * scale;
            end
        end

        function x = multSqTr(obj, y)
            if isempty(obj.FrobNorm)
                x = obj.St(y);
            else
                scale = (obj.FrobNorm^2 / (obj.M * obj.N)) * sum(y(:));
                x = ones(obj.N, 1) * scale;
            end
        end
    end

    methods (Access = private)
        function frobNorm = estimateFrobeniusNorm(obj, nProbe)
            frobNormSq = 0;
            for probeIdx = 1:nProbe %#ok<NASGU>
                x = randn(obj.N, 1);
                y = obj.mult(x);
                frobNormSq = frobNormSq + (norm(y(:), 'fro') / max(norm(x(:), 'fro'), eps))^2;
            end
            frobNorm = sqrt(frobNormSq * (obj.N / nProbe));
        end
    end
end
