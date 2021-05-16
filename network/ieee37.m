%	This code is modified from to linearize IEEE 37-bus feeder system.
%
%	S. Bolognani, F. Dörfler (2015)
%	"Fast power system analysis via implicit linearization of the power flow manifold."
%	In Proc. 53rd Annual Allerton Conference on Communication, Control, and Computing.
%	Preprint available at http://control.ee.ethz.ch/~bsaverio/papers/BolognaniDorfler_Allerton2015.pdf
%
%	This source code is distributed in the hope that it will be useful, but without any warranty.
%
%	MatLab OR GNU Octave, version 3.8.1 available at http://www.gnu.org/software/octave/
%	MATPOWER 5.1 available at http://www.pserc.cornell.edu/matpower/

clear all
close all
clc

% Load grid model
%Vbase = 4160/sqrt(3);
%Sbase = 5e6;
%Zbase = Vbase^2/Sbase;
Zbase = 1;
Vbase = 4800;
Sbase = (Vbase^2)/Zbase;

phase = 1;

[Pbus, Qbus, Ybus] = extract_phase_37feeder(phase, Zbase, Sbase);
Sbus = complex(Pbus, Qbus);
n = size(Ybus, 1); 
%%
% Compute exact solution via MatPower
ref_idx = [1];
pv_idx = [];%[4, 7, 9. 10, 11, 13, 16, 17, 20, 22, 23, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36];
pq_idx =[2:36];%[2, 3, 5, 6, 8, 12, 14, 15, 16, 18, 19, 21, 24, 25, 27];
V0 = ones(n,1);
%[results, success, i] = gausspf(Ybus, Sbus, V0, ref_idx, pv_idx, pq_idx, mpoption('VERBOSE', 1, 'OUT_ALL',0));

%%
%%%%% LINEARIZED MODEL %%%%%

%%%%% Linearization point (given voltage magnitude and angle)
%Vbus = NaN(n,1);
%Vbus(mpc.gen(:,GEN_BUS)) = mpc.gen(:,VG);
Vbus = ones(n,1);

% Flat voltage profile
V0 = ones(n,1);
A0 = zeros(n,1);

% Corresponding current injection
J0 = Ybus*(V0.*exp(1j*A0));

% Corresponding power injection
S0 = V0.*exp(1j*A0).*conj(J0);
P0 = real(S0);
Q0 = imag(S0);

%%%%% Linear system of equations for the grid model

UU = bracket(diag(V0.*exp(1j*A0)));
JJ = bracket(diag(conj(J0)));
NN = Nmatrix(2*n);
YY = bracket(Ybus);
PP = Rmatrix(ones(n,1), zeros(n,1));

AA = zeros(2*n,4*n);
BB = zeros(2*n,1);

V_OFFSET = 0;
A_OFFSET = 1*n;
P_OFFSET = 2*n;
Q_OFFSET = 3*n;

% bus models

for bus = 1:n
	row = 2*(bus-1)+1;
	if (any(bus == pq_idx(:)))
		AA(row,P_OFFSET+bus) = 1;
		AA(row+1,Q_OFFSET+bus) = 1;
        BB(row) = Pbus(bus) - P0(bus);
		BB(row+1) = Qbus(bus) - Q0(bus);
	elseif (any(bus == pv_idx(:)))
		AA(row,P_OFFSET+bus) = 1;
		AA(row+1,V_OFFSET+bus) = 1;
        BB(row) = Pbus(bus) - P0(bus);
		BB(row+1) = Vbus(bus) - V0(bus);
	elseif (any(bus == ref_idx(:)))
		AA(row,V_OFFSET+bus) = 1;
		AA(row+1,A_OFFSET+bus) = 1;
		BB(row) = Vbus(bus) - V0(bus);
        BB(row+1) = 0 - A0(bus);
	end
end

Agrid = [(JJ + UU*NN*YY)*PP -eye(2*n)];
Amat = [Agrid; AA];
Bmat = [zeros(2*n,1); BB]; 

x = Amat\Bmat;

approxVM = V0 + x(1:n);
approxVA = (A0 + x(n+1:2*n))/pi*180;


%%
% Check my implementation is correct
A11 = (JJ + UU*NN*YY)*PP;
A21 = AA(:, 1:2*n);
A22 = AA(:, 2*n+1:4*n);

n_new = n-1;

delta_P = reshape(Pbus(2:end), n_new, 1)-P0(2:end);
delta_Q = reshape(Qbus(2:end), n_new, 1)-Q0(2:end);

% remove the first bus;
A11(n+1, :) = [];
A11(:, n+1) = [];
A11(1, :) = [];
A11(:, 1) = [];

x_hat = inv(A11) * [delta_P; delta_Q];
%x_hat = pinv([A11; A21]) * ([eye(2*n); -A22] * [Pbus.'; Qbus.'] + Bmat);

myVM = V0(2:end) + x_hat(1:n_new);
myVA = (A0(2:end) + x_hat(n_new+1:2*n_new))/pi*180;

subplot(211)
%plot(1:n, approxVM, 'k*')
plot(2:n, myVM(1:end), 'ko', 1:n, approxVM, 'k*')
%plot(1:n, results.bus(:,VM), 'ko', 1:n, approxVM, 'k*')
ylabel('magnitudes [p.u.]')
xlim([0 n])

subplot(212)
%plot(1:n, approxVA, 'k*')
plot(2:n, myVA, 'ko', 1:n, approxVA, 'k*')
%plot(1:n, results.bus(:,VA), 'ko', 1:n, approxVA, 'k*')
%ylabel('angles [deg]')
xlim([0 n])
%%
H = inv(A11);
R = H(1:n_new, 1:n_new);
B = H(1:n_new, n_new+1:2*n_new);


