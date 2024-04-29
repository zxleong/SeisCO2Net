%% Basic 3D simulation of a two-phase water and gas system
% This example shows the creation of a simple 3D grid (not VE) in mrst from
% scratch and basic usage of the the TwoPhaseWaterGasModel for modelling a
% CO2-H2O system. CO2 is injected into an intially brine filled reservoir
% from an injection well at the bottom of the reservoir. We model two years
% of injection and two years post injection. CO2 properties are taken from 
% co2lab's tabulated co2props() function.

%%



%% Load modules
clear all
clf;
addpath('/mrst-2022a')
run startup.m

mrstModule add co2lab ad-core ad-props ad-blackoil mrst-gui;

%% Run through realizations
for ireal=0:39
    display(sprintf('SIMULATION REALIZATION %d NOW !',ireal))

    %% Load perm and porosity and time vec

    perm_path = 'data/mat/perm%d.mat';
    poro_path = 'data/mat/poro%d.mat';

    load(sprintf(perm_path,ireal))
    load(sprintf(poro_path,ireal))

    %% Make cartesian grid

    dx = 1.5; %m
    dy = 1.5; %m
    dz = 1.5; %m

    X = 0:dx:32*dx; 
    Y = 0:dx:48*dy; 
    Z = 0:dz:45*dz;  

    G = tensorGrid(X,Y,Z);
    G = computeGeometry(G);

    %% My own Perm/Poro

    rock.perm = perm3d(:)* 9.869233e-16 ;
    rock.poro = poro3d(:);

    %% Initial state
    gravity on; % tell MRST to turn on gravity
    g = gravity; % get the gravity vector
    rhow = 1000; % density of brine 
    initState.pressure = rhow * g(3) * G.cells.centroids(:,3); % initial pressure
    initState.s = repmat([1, 0], G.cells.num, 1); % initial saturations
    initState.sGmax = initState.s(:,2); % initial max. gas saturation (hysteresis)

    %% Fluid model
    co2     = CO2props(); % load sampled tables of co2 fluid properties
    p_ref   = 15 * mega * Pascal; % choose reference pressure
    t_ref   = 70 + 273.15; % choose reference temperature, in Kelvin
    rhoc    = co2.rho(p_ref, t_ref); % co2 density at ref. press/temp
    cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
    cf_wat  = 0; % brine compressibility (zero)
    cf_rock = 4.35e-5 / barsa; % rock compressibility
    muw     = 8e-4 * Pascal * second; % brine viscosity
    muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

    mrstModule add ad-props; % The module where initSimpleADIFluid is found

    % Use function 'initSimpleADIFluid' to make a simple fluid object
    fluid = initSimpleADIFluid('phases', 'WG'           , ...
                               'mu'  , [muw, muco2]     , ...
                               'rho' , [rhow, rhoc]     , ...
                               'pRef', p_ref            , ...
                               'c'   , [cf_wat, cf_co2] , ...
                               'cR'  , cf_rock          , ...
                               'n'   , [2 2]);

    % Change relperm curves
    srw = 0.27;
    src = 0.20;
    fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
    fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));

    % Add capillary pressure curve
    pe = 5 * kilo * Pascal;
    pcWG = @(sw) pe * sw.^(-1/2);
    fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); %@@


    %% Wells
    wc_global = false(G.cartDims);
    wc_global(16,13,29) = true;
    wc = find(wc_global(G.cells.indexMap));

    % Calculate the injection rate
    % inj_rate = 1.5 * mega * 1e3 / year / fluid.rhoGS;
    inj_rate =  1/fluid.rhoGS;
    % Start with empty set of wells
    W = [];

    % Add a well to the set
    W = addWell(W, G, rock, wc, ...
                'refDepth', G.cells.centroids(wc, 3), ... % BHP reference depth
                'type', 'rate', ...  % inject at constant rate
                'val', inj_rate, ... % volumetric injection rate
                'comp_i', [0 1]);    % inject CO2, not water

    %% Boundary conditions

    % Start with an empty set of boundary faces
    bc = [];

    % identify all vertical faces
    vface_ind = (G.faces.normals(:,3) == 0);

    % identify all boundary faces (having only one cell neighbor
    bface_ind = (prod(G.faces.neighbors, 2) == 0);

    % identify all lateral boundary faces
    bc_face_ix = find(vface_ind & bface_ind);

    % identify cells neighbouring lateral boundary baces
    bc_cell_ix = sum(G.faces.neighbors(bc_face_ix,:), 2);

    % lateral boundary face pressure equals pressure of corresponding cell
    p_face_pressure = initState.pressure(bc_cell_ix); 

    % Add hydrostatic pressure conditions to open boundary faces
    bc = addBC(bc, bc_face_ix, 'pressure', p_face_pressure, 'sat', [1, 0]);

    %% Schedule

    % Setting up two copies of the well and boundary specifications. 
    % Modifying the well in the second copy to have a zero flow rate.
    schedule.control    = struct('W', W, 'bc', bc);
    schedule.control(2) = struct('W', W, 'bc', bc);
    schedule.control(2).W.val = 0;

    dT = rampupTimesteps(5*day, 3*hour);
%     dT=diff(Time);

    schedule.step.val = dT; % 5-day Injection Period

    % Specifying which control to use for each timestep.
    % schedule.step.control = [ones(numel(dT), 1); ones(8,1)*2];
    schedule.step.control = ones(numel(dT), 1);


    %% Model
    model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);


    %% save progress simulation
    ireal_name = 'v4_f%d';
    BaseName = sprintf(ireal_name,ireal);
    mainrun = packSimulationProblem(initState, model, schedule, BaseName);
    problems = {mainrun};
    [ok, status] = simulatePackedProblem(problems); 
    [wellSol, states, reports, names, reportTime] = getMultiplePackedSimulatorOutputs(problems);


    %% Save
    sPath = 'MRST_CO2_Outputs/v4_states_f%d.mat';
    save(sprintf(sPath,ireal),'states');

end


%%
% <html>
% <p><font size="-1">
% Copyright 2009-2021 SINTEF Digital, Mathematics & Cybernetics.
% </font></p>
% <p><font size="-1">
% This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).
% </font></p>
% <p><font size="-1">
% MRST is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% </font></p>
% <p><font size="-1">
% MRST is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% </font></p>
% <p><font size="-1">
% You should have received a copy of the GNU General Public License
% along with MRST.  If not, see
% <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses</a>.
% </font></p>
% </html>


