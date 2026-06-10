classdef A_wavelet_3d_LinTrans < LinTrans
    %A_WAVELET_3D_LINTRANS 3-D wavelet transform as a LinTrans object.
    %
    %   The forward operation maps a wavelet coefficient vector to an image
    %   volume using the inverse wavelet transform.  The adjoint operation
    %   maps an image volume to its wavelet coefficient vector.

    properties
        M               % Output dimension: number of voxels
        N               % Input dimension: number of wavelet coefficients
        mat_sz          % Volume size
        wav_coef_len    % Number of wavelet coefficients
        A               % Inverse wavelet transform: struct -> volume
        Ah              % Forward wavelet transform: volume -> struct
        E               % Coefficients -> wavelet struct
        Et              % Wavelet struct -> coefficients
        S               % Optional exact squared forward operator
        St              % Optional exact squared adjoint operator
    end

    methods
        function obj = A_wavelet_3d_LinTrans(M, N, mat_sz, wav_coef_len, A, Ah, E, Et, S, St)
            obj = obj@LinTrans;

            validateattributes(M, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(N, {'numeric'}, {'scalar', 'integer', 'positive'});
            validateattributes(mat_sz, {'numeric'}, {'vector', 'numel', 3});
            validateattributes(wav_coef_len, {'numeric'}, {'scalar', 'integer', 'positive'});

            if ~isa(A, 'function_handle') || ~isa(Ah, 'function_handle') || ...
                    ~isa(E, 'function_handle') || ~isa(Et, 'function_handle')
                error('A_wavelet_3d_LinTrans:InvalidFunctionHandle', ...
                    'A, Ah, E, and Et must be function handles.');
            end

            obj.M = double(M);
            obj.N = double(N);
            obj.mat_sz = double(mat_sz(:).');
            obj.wav_coef_len = double(wav_coef_len);
            obj.A = A;
            obj.Ah = Ah;
            obj.E = E;
            obj.Et = Et;

            if obj.M ~= prod(obj.mat_sz)
                error('A_wavelet_3d_LinTrans:DimensionMismatch', ...
                    'M must equal prod(mat_sz).');
            end
            if obj.N ~= obj.wav_coef_len
                error('A_wavelet_3d_LinTrans:DimensionMismatch', ...
                    'N must equal wav_coef_len.');
            end

            if nargin >= 9 && ~isempty(S)
                if isnumeric(S) && isscalar(S) && S > 0
                    obj.FrobNorm = S;
                elseif nargin >= 10 && isa(S, 'function_handle') && isa(St, 'function_handle')
                    obj.S = S;
                    obj.St = St;
                    obj.FrobNorm = [];
                else
                    error('A_wavelet_3d_LinTrans:InvalidOptionalInputs', ...
                        'Optional inputs must be either FrobNorm or S/St function handles.');
                end
            else
                obj.FrobNorm = obj.estimateFrobeniusNorm(5);
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
                error('A_wavelet_3d_LinTrans:InputSizeMismatch', ...
                    'Input vector length must equal N.');
            end
            y = obj.A(obj.E(x(:)));
        end

        function x = multTr(obj, y)
            if numel(y) ~= obj.M
                error('A_wavelet_3d_LinTrans:OutputSizeMismatch', ...
                    'Input to multTr must contain M voxels.');
            end
            y = reshape(y, obj.mat_sz);
            x = obj.Et(obj.Ah(y));
        end

        function y = multSq(obj, x)
            if isempty(obj.FrobNorm)
                y = obj.S(x);
            else
                scale = (obj.FrobNorm^2 / (obj.M * obj.N)) * sum(x(:));
                y = ones(obj.mat_sz) * scale;
            end
        end

        function x = multSqTr(obj, y)
            if isempty(obj.FrobNorm)
                x = obj.St(y);
            else
                scale = (obj.FrobNorm^2 / (obj.M * obj.N)) * sum(y(:));
                x = ones(obj.wav_coef_len, 1) * scale;
            end
        end
    end

    methods (Access = private)
        function frobNorm = estimateFrobeniusNorm(obj, nProbe)
            frobNormSq = 0;
            for probeIdx = 1:nProbe %#ok<NASGU>
                x = randn(obj.wav_coef_len, 1);
                y = obj.mult(x);
                frobNormSq = frobNormSq + (norm(y(:), 'fro') / max(norm(x(:), 'fro'), eps))^2;
            end
            frobNorm = sqrt(frobNormSq * (obj.N / nProbe));
        end
    end
end
